import argparse
import gzip
import json
import os
import bson

from sqlitedict import SqliteDict
from common.hierarchical_logger import hlog, htrack
from typing import Optional


_SQLITE_FILE_SUFFIX = ".sqlite"


@htrack("Dumping all caches")
def dump_all_caches(cache_dir: str, output_dir: str, dry_run: bool):
    hlog(f"Opening Sqlite dir {cache_dir}")
    with os.scandir(cache_dir) as it:
        for entry in it:
            if entry.name.endswith(_SQLITE_FILE_SUFFIX) and entry.is_file():
                organization = entry.name[: -len(_SQLITE_FILE_SUFFIX)]
                dump_cache(
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    organization=organization,
                    dry_run=dry_run,
                )


@htrack("Dumping single cache")
def dump_cache(
    cache_dir: str,
    output_dir: str,
    organization: str,
    dry_run: bool,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
):
    if dry_run:
        hlog("Dry run mode, skipping dumping")
    if range_start:
        hlog(f"Start of range: {range_start}")
    if range_end:
        hlog(f"End of range: {range_end}")
    num_items = 0
    num_written = 0
    num_skipped = 0
    num_failed = 0
    cache_path = os.path.join(cache_dir, f"{organization}.sqlite")
    bson_path = os.path.join(output_dir, f"{organization}.bson.gz")
    hlog(f"Opening Sqlite cache {cache_path}")
    with SqliteDict(cache_path) as source_cache:
        with gzip.open(bson_path, "wb") as f:
            for key, value in source_cache.items():
                if not dry_run and (not range_start or num_items >= range_start):
                    try:
                        document = {"request": json.loads(key), "response": value}
                        f.write(bson.encode(document))
                        num_written += 1
                    except Exception:
                        num_failed += 1
                else:
                    num_skipped += 1
                num_items += 1
                if num_items % 1000 == 0:
                    hlog(f"Processed {num_items} items so far")
                    hlog(
                        f"Copied {num_written} and skipped {num_skipped} and failed "
                        + f"{num_failed} items from {cache_path} so far"
                    )
                if range_end and num_items >= range_end:
                    break

            hlog(f"Processed {num_items} total items from {cache_path}")
            hlog(
                f"Copied {num_written} and skipped {num_skipped} and failed {num_failed} total items from {cache_path}"
            )
    hlog(f"Finished with Sqlite cache {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Copy items from Sqlite to mongo")
    parser.add_argument("cache_dir", type=str, help="Directory for the .sqlite files")
    parser.add_argument("output_dir", type=str, help="BSON directory to copy items to")
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
        dump_all_caches(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            dry_run=bool(args.dry_run),
        )
    elif args.organization:
        dump_cache(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            organization=args.organization,
            dry_run=bool(args.dry_run),
            range_start=args.range_start,
            range_end=args.range_end,
        )
    else:
        raise ValueError("Either --all or --organization must be specified")


if __name__ == "__main__":
    main()
