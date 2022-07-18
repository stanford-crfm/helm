import argparse
from typing import Dict

from sqlitedict import SqliteDict

from common.cache import key_to_request, request_to_key
from common.hierarchical_logger import hlog, htrack

"""
Fix the `HuggingFaceClient` cache.

Usage:

    python3 scripts/one_off/fix_huggingface_client_cache.py -p prod_env/cache/huggingface.sqlite

"""


@htrack("Renaming the name of the tokenizer 'huggingface/gpt2_tokenizer_fast' to 'huggingface/gpt2'.")
def fix(cache_path: str):
    temp_cache: Dict[str, Dict] = dict()
    count: int

    with SqliteDict(cache_path) as cache:
        hlog(f"Found {len(cache)} entries at {cache_path}.")

        for i, (key, response) in enumerate(cache.items()):
            count = i + 1
            request: Dict = key_to_request(key)
            if request["tokenizer"] == "huggingface/gpt2_tokenizer_fast":
                # Change the name and add to `cache_copy`
                request["tokenizer"] = "huggingface/gpt2"
                new_key: str = request_to_key(request)
                temp_cache[new_key] = response

                # Delete entry with the old tokenizer name
                del cache[key]

                if count > 0 and count % 10_000 == 0:
                    hlog(f"Processed {count} entries.")
                    cache.commit()

        hlog("Copying values from temp cache...")
        for i, (key, response) in enumerate(temp_cache.items()):
            count = i + 1
            cache[key] = response

            if count > 0 and count % 10_000 == 0:
                hlog(f"Wrote {count} entries.")
                cache.commit()

        cache.commit()
        hlog(f"Still have {len(cache)} entries at {cache_path}.")


def main():
    fix(args.cache_path)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--cache-path", type=str, help="Path to cache.")
    args = parser.parse_args()

    main()
