import argparse

from pymongo import MongoClient

from helm.common.cache import create_key_value_store, MongoCacheConfig
from helm.common.hierarchical_logger import hlog, htrack, htrack_block

"""
Fix the Together cache. For each entry in the cache:
- Adds "request_type": "language-model-inference"
- Renames "engine" to "model": adds "model" and removes "engine"

Example usage:

    python3 scripts/cache/fix_together_cache.py --mongo-uri <Mongo URI of your server>

"""


@htrack("Fixing Together cache")
def fix(mongo_uri: str):
    source_name: str = "together"
    target_name: str = "together_rewritten"

    source_config = MongoCacheConfig(mongo_uri, collection_name=source_name)
    target_config = MongoCacheConfig(mongo_uri, collection_name=target_name)

    source_store = create_key_value_store(source_config)
    target_store = create_key_value_store(target_config)

    db: MongoClient = MongoClient(mongo_uri)
    if "request_1" not in db["crfm-models"][target_name].index_information():  # type: ignore
        db["crfm-models"][target_name].create_index("request", name="request_1", unique=True)

    for i, (request, response) in enumerate(source_store.get_all()):
        request["request_type"] = "language-model-inference"
        request["model"] = request["engine"]
        del request["engine"]
        target_store.put(request, response)

        if i + 1 % 10_000 == 0:
            hlog(f"Processed {i+1} entries.")

    with htrack_block(f"Dropping {source_name} collection and renaming {target_name} to {source_name}"):
        source_collection = db["crfm-models"][source_name]
        source_collection.drop()

        target_collection = db["crfm-models"][target_name]
        target_collection.rename(source_name)


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
