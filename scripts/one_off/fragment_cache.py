import argparse
import json
import os
from typing import Dict

from sqlitedict import SqliteDict

from common.hierarchical_logger import hlog, htrack

"""
Break apart the caches to smaller caches specific to the model or tokenizer.
Still keeps the old cache file around.

Run:
python3 scripts/one_off/fragment_cache.py openai
python3 scripts/one_off/fragment_cache.py ai21
python3 scripts/one_off/fragment_cache.py cohere
python3 scripts/one_off/fragment_cache.py gooseai
"""


@htrack("Fragmenting organization cache")
def fragment_cache(organization: str, output_path: str):
    old_cache_path: str = os.path.join(output_path, f"{organization}.sqlite")
    with SqliteDict(old_cache_path) as old_cache:
        for i, (key, response) in enumerate(old_cache.items()):
            request: Dict = json.loads(key)
            new_cache_file: str
            if "engine" in request:
                new_cache_file = f"{organization}-{request['engine']}.sqlite"
            else:
                tokenizer_name: str
                if organization == "cohere":
                    tokenizer_name = "cohere"
                elif organization == "ai21":
                    tokenizer_name = "j1"
                else:
                    hlog(f"WARNING: {organization} shouldn't have a tokenizer")

                new_cache_file = f"{organization}-{tokenizer_name}-tokenizer.sqlite"

            new_cache_path: str = os.path.join(output_path, new_cache_file)
            with SqliteDict(new_cache_path) as new_cache:
                new_cache[key] = response
                new_cache.commit()
                hlog(f"Request #{i}: Written out to {new_cache_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("organization", type=str, help="Name of the organization e.g., openai.")
    parser.add_argument("--output-path", type=str, default="prod_env/cache", help="Path to caches.")
    args = parser.parse_args()

    fragment_cache(args.organization, args.output_path)
    hlog("Done.")
