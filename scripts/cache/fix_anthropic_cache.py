import argparse
import time
from typing import Dict, List

from helm.common.cache import create_key_value_store, MongoCacheConfig
from helm.common.general import parse_hocon
from helm.common.hierarchical_logger import hlog, htrack
from helm.proxy.clients.anthropic_client import AnthropicClient


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

            # We've renamed "logprobs" to "logprobs
            # Compute log probs for all the entries where echo_prompt=False
            if request["echo_prompt"]:
                response["logprobs"] = response.pop("logprobs_response", None)
            else:
                logprobs_response = response.pop("logprobs_response", None)
                tokens: List[str] = logprobs_response["tokens"]
                logprobs = client.make_logprobs_request(
                    request["prompt"] + response["text"], top_k_per_token=request["k"], model_engine=request["engine"]
                )
                for key in AnthropicClient.LOGPROBS_RESPONSE_KEYS:
                    # This is a naive approach where we just take the last k tokens and log probs,
                    # where k is the number of tokens in the completion. Ideally, log probs would
                    # be included as part of the response for the inference endpoint.
                    logprobs[key] = logprobs[key][-len(tokens):]
                response["logprobs"] = logprobs
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
