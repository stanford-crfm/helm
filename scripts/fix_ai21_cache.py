import argparse

from sqlitedict import SqliteDict

from common.hierarchical_logger import hlog, htrack

"""
Script to fix an existing AI21 cache by removing unwanted entries.
Usage:
    python3 scripts/fix_ai21_cache.py -p prod_env/cache/ai21.sqlite
"""

QUOTA_EXCEEDED_ERROR = "Quota exceeded."


@htrack("AI21 cache - Removing entries with errors")
def remove_entries_with_errors(cache_path: str, dry_run: bool):
    with SqliteDict(cache_path) as cache:
        hlog(f"Found {len(cache)} entries at {cache_path}.")

        for key, response in cache.items():
            if ("detail" in response and response["detail"] == QUOTA_EXCEEDED_ERROR) or "Error" in response:
                hlog(f"Deleting entry: {response}")
                del cache[key]

        if not dry_run:
            # Write to SQLite
            cache.commit()
            hlog(f"Wrote to {cache_path}.")


def main():
    remove_entries_with_errors(args.cache_path, args.dry_run)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--cache-path", type=str, default="prod_env/cache/ai21.sqlite", help="Path to AI21 cache.",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips persisting fixes and just outputs log messages.",
    )
    args = parser.parse_args()

    main()
