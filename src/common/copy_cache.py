import argparse
import os

from sqlitedict import SqliteDict
from common.cache import _MongoKeyValueStore
from common.hierarchical_logger import hlog, htrack


_SQLITE_FILE_SUFFIX = ".sqlite"


@htrack("Copying all caches")
def copy_all_caches(cache_dir: str, mongo_host: str, dry_run: bool):
    hlog(f"Opening Sqlite dir {cache_dir}")
    with os.scandir(cache_dir) as it:
        for entry in it:
            if entry.name.endswith(_SQLITE_FILE_SUFFIX) and entry.is_file():
                organization = entry.name[: -len(_SQLITE_FILE_SUFFIX)]
                copy_cache(cache_dir, mongo_host, organization, dry_run)


@htrack("Copying single cache")
def copy_cache(cache_dir: str, mongo_host: str, organization: str, dry_run: bool):
    num_items = 0
    cache_path = os.path.join(cache_dir, f"{organization}.sqlite")
    hlog(f"Opening Sqlite cache {cache_path}")
    with SqliteDict(cache_path) as source_cache:
        with _MongoKeyValueStore(mongo_host, collection_name=organization) as target_cache:
            if dry_run:
                hlog("Dry run mode, skipping writing to mongo")
            for key, value in source_cache.items():
                if not dry_run:
                    target_cache.put(key, value)
                num_items += 1
                if num_items % 1000 == 0:
                    hlog(f"Copied {num_items} items so far")
            hlog(f"Copied {num_items} total items from {cache_path}")
    hlog(f"Finished with Sqlite cache {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Copy items from Sqlite to mongo")
    parser.add_argument("cache_dir", type=str, help="Directory for the .sqlite files")
    parser.add_argument("mongo_host", type=str, help="Mongo host to copy items to")
    parser.add_argument("--organization", type=str, help="Organization to copy cache for")
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
    if args.all:
        copy_all_caches(args.cache_dir, args.mongo_host, bool(args.dry_run))
    elif args.organization:
        copy_cache(args.cache_dir, args.mongo_host, args.organization, bool(args.dry_run))
    else:
        raise ValueError("Either --all or --organization must be specified")


if __name__ == "__main__":
    main()
