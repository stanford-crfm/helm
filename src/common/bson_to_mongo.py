import argparse
import bson
from bson import raw_bson
import gzip
import os

from common.hierarchical_logger import hlog, htrack
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


_BSON_GZ_FILE_SUFFIX = ".bson.gz"


@htrack("Copying all caches")
def copy_all_caches(bson_dir: str, mongo_uri: str, dry_run: bool):
    hlog(f"Opening BSON dir {bson_dir}")
    with os.scandir(bson_dir) as it:
        for entry in it:
            if entry.name.endswith(_BSON_GZ_FILE_SUFFIX) and entry.is_file():
                organization = entry.name[: -len(_BSON_GZ_FILE_SUFFIX)]
                copy_cache(
                    bson_dir=bson_dir,
                    mongo_uri=mongo_uri,
                    organization=organization,
                    dry_run=dry_run,
                )


@htrack("Copying single cache")
def copy_cache(
    bson_dir: str,
    mongo_uri: str,
    organization: str,
    dry_run: bool,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
):
    if dry_run:
        hlog("Dry run mode, skipping writing to mongo")
    if range_start:
        hlog(f"Start of range: {range_start}")
    if range_end:
        hlog(f"End of range: {range_end}")
    num_items = 0
    num_written = 0
    num_skipped = 0
    num_duplicates = 0
    num_failed = 0
    bson_path = os.path.join(bson_dir, f"{organization}{_BSON_GZ_FILE_SUFFIX}")
    codec_options = bson.CodecOptions(document_class=raw_bson.RawBSONDocument)
    hlog(f"Opening BSON file {bson_path}")
    mongodb_client: MongoClient = MongoClient(mongo_uri)
    database = mongodb_client.get_default_database()
    collection = database.get_collection(organization)
    for document in bson.decode_file_iter(gzip.open(bson_path, "rb"), codec_options=codec_options):  # type: ignore
        if not dry_run and (not range_start or num_items >= range_start):
            try:
                collection.insert_one(document)
                num_written += 1
            except DuplicateKeyError:
                num_duplicates += 1
            except Exception as e:
                hlog(e)
                num_failed += 1
            num_items += 1
        if num_items % 1000 == 0:
            hlog(f"Processed {num_items} items so far")
            hlog(
                f"Copied {num_written} and "
                + f"duplicated {num_duplicates} and "
                + f"skipped {num_skipped} and "
                + f"failed {num_failed} "
                + f"items from {bson_path} so far"
            )
        if range_end and num_items >= range_end:
            break
    hlog(f"Processed {num_items} total items from {bson_path}")
    hlog(
        f"Copied {num_written} and "
        + f"duplicated {num_duplicates} and "
        + f"skipped {num_skipped} and "
        + f"failed {num_failed} "
        + f"items from {bson_path} in total"
    )
    hlog(f"Finished with BSON file {bson_path}")


def main():
    parser = argparse.ArgumentParser(description="Copy items from BSON to mongo")
    parser.add_argument("bson_dir", type=str, help="BSON directory to copy items to")
    parser.add_argument("mongo_uri", type=str, help="Mongo host to copy items to")
    parser.add_argument("--organization", type=str, help="Organization to copy cache for")
    parser.add_argument("--range-start", type=int, help="The start of the range to copy")
    parser.add_argument("--range-end", type=int, help="The end of the range to copy (exclusive)")
    parser.add_argument(
        "--all",
        action="store_true",
        default=None,
        help="Copy caches for all organizations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips actually writing to mongo",
    )
    args = parser.parse_args()

    if (args.range_start or args.range_end) and not args.organization:
        raise ValueError("--range_start and --range_end require --organization to be specified")

    if args.all:
        copy_all_caches(
            bson_dir=args.bson_dir,
            mongo_uri=args.mongo_uri,
            dry_run=bool(args.dry_run),
        )
    elif args.organization:
        copy_cache(
            bson_dir=args.bson_dir,
            mongo_uri=args.mongo_uri,
            organization=args.organization,
            dry_run=bool(args.dry_run),
            range_start=args.range_start,
            range_end=args.range_end,
        )
    else:
        raise ValueError("Either --all or --organization must be specified")


if __name__ == "__main__":
    main()
