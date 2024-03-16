# This script is used to find out the max_prompt_length and max_prompt_length_plus_tokens for a given model.
# You must set max_attempts to 1 in retry.py to make it work.
# Example usage:
# python compute_request_limits.py --model_deployment_name="writer/palmyra-base" --tokenizer_name="Writer/palmyra-base"

from typing import Any, Optional, Dict
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists
from helm.clients.auto_client import AutoClient
from helm.benchmark.model_deployment_registry import ModelDeployment, get_model_deployment
from helm.tokenizers.auto_tokenizer import AutoTokenizer
from helm.common.request import Request
from helm.common.tokenization_request import TokenizationRequest

# TODO #1592: reenable this once the imports are faster
# from helm.clients.client import Client
from helm.tokenizers.tokenizer import Tokenizer

import os
import math
from tqdm import tqdm
import argparse
from attr import dataclass


def get_credentials(path: str) -> Dict[str, str]:
    # Reads the credentials from the given path
    with open(path, "r") as f:
        # Read line by line, replaces the spaces, splits on the first ":"
        # The first part is the key, the second part contians the value in between quotes
        credentials = {}
        for line in f.readlines():
            elt = line.replace(" ", "").replace("\n", "").split(":")
            if len(elt) == 2:
                credentials[elt[0]] = elt[1].split('"')[1]
        return credentials


def get_number_of_tokens(prompt: str, tokenizer: Tokenizer, tokenizer_name: str) -> int:
    tokenization_request = TokenizationRequest(tokenizer=tokenizer_name, text=prompt, encode=True)
    tokenization_response = tokenizer.tokenize(tokenization_request)
    return len(tokenization_response.tokens)


def try_request(
    client: Any,
    model_deployment_name: str,
    model_name: str,
    tokenizer_name: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    num_tokens: int,
    prefix: str = "",
    suffix: str = "",
) -> bool:
    """
    Try to make a request with the given sequence_length and num_tokens.
    Return True if the request was successful, False otherwise.
    """
    num_tokens_prefix = get_number_of_tokens(prefix, tokenizer, tokenizer_name)
    num_tokens_suffix = get_number_of_tokens(suffix, tokenizer, tokenizer_name)

    try:
        request = Request(
            model=model_name,
            model_deployment=model_deployment_name,
            prompt=prefix + " ".join(["hello"] * (sequence_length - num_tokens_prefix - num_tokens_suffix)) + suffix,
            max_tokens=num_tokens,
        )
        response = client.make_request(request)
        return response.success
    except Exception:
        return False


@dataclass
class RequestLimits:
    max_prompt_length: int
    num_tokens_prefix: int
    num_tokens_suffix: int
    usable_max_prompt_length: int
    max_prompt_length_plus_tokens: Optional[int] = None


def figure_out_max_prompt_length(
    client: AutoClient,
    auto_tokenizer: AutoTokenizer,
    model_deployment_name: str,
    model_name: str,
    tokenizer_name: str,
    upper_bound: int = 9500,
    lower_bound: int = 450,
    prefix: str = "",
    suffix: str = "",
) -> RequestLimits:
    tokenizer = auto_tokenizer._get_tokenizer(tokenizer_name)
    num_tokens_prefix = get_number_of_tokens(prefix, tokenizer, tokenizer_name)
    num_tokens_suffix = get_number_of_tokens(suffix, tokenizer, tokenizer_name)

    # Perform a binary search to find the max tokens between lower_bound and upper_bound
    lower_bound += num_tokens_prefix + num_tokens_suffix
    pbar: tqdm
    with tqdm(total=int(math.log2(upper_bound - lower_bound))) as pbar:
        while lower_bound < upper_bound:
            middle = math.ceil((lower_bound + upper_bound) / 2)
            if try_request(
                client, model_deployment_name, model_name, tokenizer_name, tokenizer, middle, 0, prefix, suffix
            ):
                lower_bound = middle
            else:
                upper_bound = middle - 1
            pbar.update(1)

    # Just in case the number of tokens does not match the number of words, check number of tokens with tokenizer
    max_prompt_length = get_number_of_tokens(
        prefix + " ".join(["hello"] * (lower_bound - num_tokens_prefix - num_tokens_suffix)) + suffix,
        tokenizer,
        tokenizer_name,
    )
    return RequestLimits(
        max_prompt_length=max_prompt_length,
        num_tokens_prefix=num_tokens_prefix,
        num_tokens_suffix=num_tokens_suffix,
        usable_max_prompt_length=lower_bound,
    )


def figure_out_max_prompt_length_plus_tokens(
    client: AutoClient,
    auto_tokenizer: AutoTokenizer,
    model_deployment_name: str,
    model_name: str,
    tokenizer_name: str,
    max_prompt_length: int,
    prefix: str = "",
    suffix: str = "",
) -> int:
    tokenizer = auto_tokenizer._get_tokenizer(tokenizer_name)
    lower_bound = 1
    upper_bound = 2 * max_prompt_length + 1

    # Check if there is a limit (some model accept as many tokens as you want)
    if try_request(
        client,
        model_deployment_name,
        model_name,
        tokenizer_name,
        tokenizer,
        max_prompt_length,
        2**31 - 2 - max_prompt_length,
        prefix,
        suffix,
    ):
        print("The model has no limit on the number of tokens")
        return -1
    else:
        print("The model has a limit on the number of tokens")

    # Perform a binary search to find the max tokens between lower_bound and upper_bound
    pbar: tqdm
    with tqdm(total=int(math.log2(upper_bound - lower_bound))) as pbar:
        while lower_bound < upper_bound:
            middle = math.ceil((lower_bound + upper_bound) / 2)
            if try_request(
                client,
                model_deployment_name,
                model_name,
                tokenizer_name,
                tokenizer,
                max_prompt_length,
                middle,
                prefix,
                suffix,
            ):
                lower_bound = middle
            else:
                upper_bound = middle - 1
            pbar.update(1)

    return lower_bound + max_prompt_length


def check_limits(
    client: AutoClient,
    auto_tokenizer: AutoTokenizer,
    model_deployment_name: str,
    model_name: str,
    tokenizer_name: str,
    limits: RequestLimits,
    prefix: str = "",
    suffix: str = "",
) -> bool:
    tokenizer = auto_tokenizer._get_tokenizer(tokenizer_name)
    result: bool = True

    # Check the max_prompt_length
    max_prompt_length = limits.max_prompt_length
    if max_prompt_length < 0:
        print("No limit on the number of tokens")
        if not try_request(
            client, model_deployment_name, model_name, tokenizer_name, tokenizer, 2**32 - 2, 0, prefix, suffix
        ):
            print(f"There is a limit on the number of tokens. Params: max_prompt_length={2**32 - 2}, max_tokens=1")
            result = False
    else:
        # There is a limit on the number of tokens
        # If there is no limit on the number of tokens, max_prompt_length should be -1
        # And we should not be here
        # Check that max_prompt_length is ok
        if not try_request(
            client, model_deployment_name, model_name, tokenizer_name, tokenizer, max_prompt_length, 0, prefix, suffix
        ):
            print(f"max_prompt_length is too big. Params: max_prompt_length={max_prompt_length}, max_tokens=1")
            result = False
        # Check that max_prompt_length + 1 is not ok
        if try_request(
            client,
            model_deployment_name,
            model_name,
            tokenizer_name,
            tokenizer,
            max_prompt_length + 1,
            0,
            prefix,
            suffix,
        ):
            print(f"max_prompt_length could be bigger. Params: max_prompt_length={max_prompt_length+1}, max_tokens=1")
            result = False
        # Check that max_prompt_length - 1 is ok
        if not try_request(
            client,
            model_deployment_name,
            model_name,
            tokenizer_name,
            tokenizer,
            max_prompt_length - 1,
            0,
            prefix,
            suffix,
        ):
            print(
                f"max_prompt_length ssems to be inconsistent. max_prompt_length={max_prompt_length} "
                f"is ok but max_prompt_length={max_prompt_length-1} is not, with max_tokens=0"
            )
            result = False

    # Check the max_prompt_length_plus_tokens
    max_prompt_length_plus_tokens = limits.max_prompt_length_plus_tokens
    if max_prompt_length_plus_tokens is None:
        print("Setting max_prompt_length_plus_tokens max_prompt_length as it was not provided")
        max_prompt_length_plus_tokens = max_prompt_length
    if max_prompt_length_plus_tokens < 0:
        print("No limit on the number of tokens")
        if not try_request(
            client,
            model_deployment_name,
            model_name,
            tokenizer_name,
            tokenizer,
            max(1, max_prompt_length),
            2**32 - 2,
            prefix,
            suffix,
        ):
            print(
                f"There is a limit on the number of tokens. Params: max_prompt_length={max_prompt_length},"
                f" max_tokens={2**32 - 2}"
            )
            result = False
    else:
        # There is a limit on the number of tokens
        # Check that max_prompt_length_plus_tokens is ok
        # If there is no limit on the number of tokens, we skip this test
        if not try_request(
            client,
            model_deployment_name,
            model_name,
            tokenizer_name,
            tokenizer,
            max_prompt_length,
            max_prompt_length_plus_tokens - max_prompt_length,
            prefix,
            suffix,
        ):
            print(
                f"max_prompt_length_plus_tokens is too big. Params: max_prompt_length={max_prompt_length},"
                f" max_tokens={max_prompt_length_plus_tokens-max_prompt_length}"
            )
            result = False
        if try_request(
            client,
            model_deployment_name,
            model_name,
            tokenizer_name,
            tokenizer,
            max_prompt_length,
            max_prompt_length_plus_tokens - max_prompt_length + 1,
            prefix,
            suffix,
        ):
            print(
                f"max_prompt_length_plus_tokens could be bigger. Params: max_prompt_length={max_prompt_length},"
                f" max_tokens={max_prompt_length_plus_tokens-max_prompt_length+1}"
            )
            result = False

    return result


def get_args():
    # model_name, tokenizer_name, prefix and suffix are passed as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_deployment_name", type=str, default="writer/palmyra-base")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="Writer/palmyra-base")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help='The prefix to use before the prompt. For example for anthropic, use "Human: "',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help='The suffix to use after the prompt. For example for anthropic, use "Assistant: "',
    )
    parser.add_argument("--credentials_path", type=str, default="../prod_env/credentials.conf")
    parser.add_argument("--cache_path", type=str, default="../prod_env/cache")
    args = parser.parse_args()

    if args.model_name == "":
        model_deployment: ModelDeployment = get_model_deployment(args.model_deployment_name)
        args.model_name = model_deployment.model_name
    return args


def main():
    print("Imports Done\n")
    args = get_args()

    credentials_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.credentials_path)
    credentials = get_credentials(credentials_path)

    print("=========== Initial setup ===========")
    current_path = os.path.dirname(os.path.realpath(__file__))
    cache_path = os.path.join(current_path, args.cache_path)
    print(f"cache_path: {cache_path}")

    ensure_directory_exists(cache_path)
    client = AutoClient(
        credentials=credentials, file_storage_path=cache_path, cache_backend_config=SqliteCacheBackendConfig(cache_path)
    )
    auto_tokenizer = AutoTokenizer(credentials=credentials, cache_backend_config=SqliteCacheBackendConfig(cache_path))
    print("client successfully created")

    print("Making short request...")
    request = Request(
        model=args.model_name,
        model_deployment=args.model_deployment_name,
        prompt=args.prefix + "hello" + args.suffix,
        max_tokens=1,
    )
    response = client.make_request(request)
    if not response.success:
        raise ValueError("Request failed")
    print("Request successful")
    print("=====================================")
    print("")

    print("========== Model infos ==========")
    print(f"model_name: {args.model_name}")
    print(f"tokenizer_name: {args.tokenizer_name}")
    print(f"prefix: {args.prefix}")
    print(f"suffix: {args.suffix}")
    print("=================================")
    print("")

    print("========== Figure out max_prompt_length ==========")
    limits: RequestLimits = figure_out_max_prompt_length(
        client,
        auto_tokenizer,
        args.model_deployment_name,
        args.model_name,
        args.tokenizer_name,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    print(f"max_prompt_length: {limits.max_prompt_length}")
    print("===================================================")
    print("")

    print("========== Figure out max_prompt_length_plus_tokens ==========")
    max_prompt_length_plus_tokens: int = figure_out_max_prompt_length_plus_tokens(
        client,
        auto_tokenizer,
        args.model_deployment_name,
        args.model_name,
        args.tokenizer_name,
        max_prompt_length=limits.max_prompt_length,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    limits.max_prompt_length_plus_tokens = max_prompt_length_plus_tokens
    print(f"max_prompt_length_plus_tokens: {limits.max_prompt_length_plus_tokens}")
    print("==============================================================")
    print("")

    # Check the limits
    print("========== Check the limits ==========")
    result: bool = check_limits(
        client,
        auto_tokenizer,
        args.model_deployment_name,
        args.model_name,
        args.tokenizer_name,
        limits,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    if result:
        print("All limits are respected")
    else:
        print("Some limits are not respected")
    print("======================================")
    print("")

    # Print the infos
    print("========== Print the limits ==========")
    for key, value in limits.__dict__.items():
        print(f"{key}: {value}")
    print("=====================================")


if __name__ == "__main__":
    main()
