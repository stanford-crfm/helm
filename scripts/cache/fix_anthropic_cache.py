import argparse
import time
from typing import Dict

from common.cache import create_key_value_store, MongoCacheConfig
from common.general import parse_hocon
from common.hierarchical_logger import hlog, htrack
from proxy.clients.anthropic_client import AnthropicClient

"""
Adds log probs for all entries in the Anthropic cache.

Example usage:

    python3 scripts/cache/fix_anthropic_cache.py --mongo-uri <MongoDB URI>
"""

DEFAULT_TOP_K: int = 1


@htrack("Updating Anthropic cache with logprobs.")
def add_logprobs(mongo_uri: str, credentials_path: str):
    with open(credentials_path) as f:
        credentials: Dict[str, str] = parse_hocon(f.read())
        api_key: str = credentials["anthropicApiKey"]

    cache_config = MongoCacheConfig(mongo_uri, collection_name="anthropic")
    client = AnthropicClient(api_key, cache_config)

    with create_key_value_store(cache_config) as cache:
        for i, (request, response) in enumerate(cache.get_all()):
            # Send and time how long the logprobs request takes. Add the time to `request_time`
            start: float = time.time()

            if "logprobs" in response:
                continue

            # Compute log probs for all the entries where echo_prompt=False
            if request["echo_prompt"]:
                response["logprobs"] = response.pop("logprobs_response", None)
            else:
                response["logprobs"] = client.make_logprobs_request(
                    request["prompt"] + response["text"], top_k_per_token=request["k"], model_engine=request["engine"]
                )
                response.pop("logprobs_response", None)
                response.pop("skipped_logprobs_request", None)

            request_time: float = time.time() - start
            response["request_time"] += request_time

            # We've only updated the responses at this point.
            cache.put(request, response)
            hlog(f"Processed entry #{i + 1} ({request_time:.2f}s)")


def main():
    add_logprobs(args.mongo_uri, args.credentials_path)
    hlog("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--credentials-path", type=str, default="prod_env/credentials.conf", help="Path to credentials."
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help=(
            "For a MongoDB cache, the Mongo URI. " "Example format: mongodb://[username:password@]host1[:port1]/dbname"
        ),
    )
    args = parser.parse_args()

    main()
