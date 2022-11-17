import argparse

from helm.common.cache import create_key_value_store, MongoCacheConfig
from helm.common.hierarchical_logger import hlog, htrack

"""
Removes Together API entries from cache.

Example usage:

    python3 scripts/cache/remove_together_api_entries.py --mongo-uri <Mongo URI of your server>

"""


@htrack("Removing Together entries")
def fix(mongo_uri: str):
    count: int = 0
    cache_config = MongoCacheConfig(mongo_uri, collection_name="together")

    with create_key_value_store(cache_config) as store:
        for i, (request, response) in enumerate(store.get_all()):
            # Responses from the API have "request_time" set to a float,
            # while batch requests have batch information in their responses.
            if type(response["request_time"]) == float:
                store.remove(request)
                count += 1

            if (i + 1) % 10_000 == 0:
                hlog(f"Processed {i+1} entries")

    hlog(f"Removed {count} entries.")


def main():
    fix(args.mongo_uri)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help=(
            "For a MongoDB cache, Mongo URI to update. "
            "Example format: mongodb://[username:password@]host1[:port1]/dbname"
        ),
    )
    args = parser.parse_args()

    main()
