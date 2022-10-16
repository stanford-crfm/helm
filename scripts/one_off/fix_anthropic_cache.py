import argparse
import json
import time
from typing import Dict, List

from sqlitedict import SqliteDict

from common.cache import key_to_request, request_to_key, CacheConfig
from common.general import parse_hocon
from common.hierarchical_logger import hlog, htrack
from proxy.clients.anthropic_client import AnthropicClient

"""
Updates the entries of an old Anthropic cache with logprobs.
Running with `--light` skips making logprobs requests.

Usage:

    python3 scripts/one_off/fix_anthropic_cache.py --light

"""

DEFAULT_TOP_K: int = 1


@htrack("Updating Anthropic cache with logprobs.")
def fix(cache_path: str, credentials_path: str, light: bool):
    with open(credentials_path) as f:
        credentials: Dict[str, str] = parse_hocon(f.read())
        api_key: str = credentials["anthropicApiKey"]

    new_cache: Dict[str, Dict] = dict()
    client = AnthropicClient(api_key, CacheConfig(cache_path))
    with SqliteDict(cache_path) as cache:
        num_entries: int = len(cache)
        hlog(f"Found {num_entries} entries at {cache_path}.")

        for i, (key, response) in enumerate(cache.items()):
            # Handle request:
            # - Update k=-1 -> k=1 (the default value of k)
            # - Add echo_prompt=False (echo_prompt was not supported before)
            request: Dict = key_to_request(key)
            request["k"] = DEFAULT_TOP_K
            request["echo_prompt"] = False
            new_key: str = request_to_key(request)

            # Handle responses:
            # - Extract completion and set `text` to it
            # - remove `tokens` and `raw_response`
            # - Make top k logprobs request and save it as `logprobs_response` if not light
            completion_response = json.loads(response["raw_response"])
            completion: str = completion_response["completion"]
            response["text"] = completion
            tokens: List[str] = response["tokens"]
            del response["tokens"]
            del response["raw_response"]

            if light:
                # Mark this entry, so we can add logprobs later
                response["skipped_logprobs_request"] = True
                num_tokens: int = len(tokens)
                logprobs_response = {
                    "tokens": tokens,
                    # These are just placeholders
                    "logprobs": [0] * num_tokens,
                    "topk_logprobs": [[]] * num_tokens,
                    "topk_tokens": [[]] * num_tokens,
                }
                response["logprobs_response"] = json.dumps(logprobs_response)
                hlog(f"Entry #{i + 1} out of {num_entries}")
            else:
                # Send and time how long the logprobs request takes. Add the time to `request_time`
                start: float = time.time()
                response["logprobs_response"] = client.make_logprobs_request(
                    completion, top_k_per_token=DEFAULT_TOP_K, model_engine=request["engine"]
                )
                request_time: float = time.time() - start
                response["request_time"] += request_time
                hlog(f"Entry #{i+1} out of {num_entries} ({request_time:.2f}s)")

            # Add the updated entry
            new_cache[new_key] = response

            # Delete the old entry
            del cache[key]

        # Copy over the new entries
        for key, response in new_cache.items():
            cache[key] = response
            cache.commit()

        hlog(f"Updated {len(cache)} entries.")


def main():
    fix(args.cache_path, args.credentials_path, args.light)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--cache-path", type=str, default="prod_env/cache/anthropic.sqlite", help="Path to cache."
    )
    parser.add_argument(
        "-c", "--credentials-path", type=str, default="prod_env/credentials.conf", help="Path to credentials."
    )
    parser.add_argument(
        "--light",
        action="store_true",
        default=None,
        help="Skips making logprobs request",
    )
    args = parser.parse_args()

    main()
