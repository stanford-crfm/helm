import argparse
import json
import time
from typing import Any, Callable, Dict, List

from helm.common.cache import create_key_value_store, MongoCacheConfig
from helm.common.general import parse_hocon
from helm.common.hierarchical_logger import hlog, htrack
from helm.clients.anthropic_client import AnthropicLegacyClient
from helm.proxy.retry import get_retry_decorator


"""
Adds log probs for all entries in the Anthropic cache.

Example usage:

    python3 scripts/cache/fix_anthropic_cache.py --mongo-uri <MongoDB URI>
"""

DEFAULT_TOP_K: int = 1


def retry_if_request_failed(result: Any) -> bool:
    if type(result) is not dict:
        hlog(f"Unexpected response: {result}")

    return type(result) is not dict


retry_request: Callable = get_retry_decorator(
    "Request", max_attempts=10, wait_exponential_multiplier_seconds=5, retry_on_result=retry_if_request_failed
)


@retry_request
def make_logprobs_request_with_retry(
    client: AnthropicLegacyClient, text: str, top_k_per_token: int, model_engine: str
) -> Dict[str, List[str]]:
    return client.make_logprobs_request(text, top_k_per_token, model_engine)


@htrack("Updating Anthropic cache with logprobs.")
def add_logprobs(mongo_uri: str, credentials_path: str, dry_run: bool):
    with open(credentials_path) as f:
        credentials: Dict[str, str] = parse_hocon(f.read())
        api_key: str = credentials["anthropicApiKey"]

    cache_config = MongoCacheConfig(mongo_uri, collection_name="anthropic")
    client = AnthropicLegacyClient(api_key=api_key, cache_config=cache_config)

    with create_key_value_store(cache_config) as cache:
        for i, (request, response) in enumerate(cache.get_all()):
            if "logprobs" in response:
                hlog(f"This entry was already updated: {response}")
                continue

            process_time: float
            logprobs_response = json.loads(response.pop("logprobs_response", None))
            if request["echo_prompt"]:
                process_time = 0
                # We've renamed "logprobs_response" to "logprobs
                response["logprobs"] = logprobs_response
            else:
                tokens: List[str] = logprobs_response["tokens"]

                # Compute log probs for all the entries where echo_prompt=False
                # Send and time how long the logprobs request takes. Add the time to `request_time`
                start: float = time.time()
                logprobs = make_logprobs_request_with_retry(
                    client,
                    text=request["q"] + "".join(tokens),
                    top_k_per_token=request["k"],
                    model_engine=request["engine"],
                )
                request_time: float = time.time() - start
                response["request_time"] += request_time
                process_time = request_time

                for key in AnthropicLegacyClient.LOGPROBS_RESPONSE_KEYS:
                    # This is a naive approach where we just take the last k tokens and log probs,
                    # where k is the number of tokens in the completion. Ideally, log probs would
                    # be included as part of the response for the inference endpoint.
                    logprobs[key] = logprobs[key][-len(tokens) :]
                response["logprobs"] = logprobs
                response["check_logprobs"] = tokens != logprobs["tokens"]
                response.pop("skipped_logprobs_request", None)

            hlog(f"Processed entry #{i + 1} ({process_time:.2f}s)")
            if dry_run:
                hlog(f"[DRY] Updating cache with\nrequest: {request}\nresponse: {response}")
            else:
                # We've only updated the responses at this point.
                cache.put(request, response)


def main():
    add_logprobs(args.mongo_uri, args.credentials_path, args.dry_run)
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips updating the cache and just logs instead.",
    )
    args = parser.parse_args()

    main()
