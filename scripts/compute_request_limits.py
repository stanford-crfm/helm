# This script is used to find out the max_prompt_length and max_prompt_length_plus_tokens for a given model.
# You must set max_attempts to 1 in retry.py to make it work.
# Example usage:
# python compute_request_limits.py --model_name="writer/palmyra-base" --tokenizer_name="Writer/palmyra-base"

from typing import Any, Optional, Dict
from helm.proxy.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.common.tokenization_request import TokenizationRequest

# Only used for typing and slow to import, so removed
# from helm.proxy.clients.client import Client

import os
import math
import random
from tqdm import tqdm
import argparse

print("Imports Done\n")

# model_name, tokenizer_name, prefix and suffix are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="writer/palmyra-base")
parser.add_argument("--tokenizer_name", type=str, default="Writer/palmyra-base")
parser.add_argument("--prefix", type=str, default="")
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--relative_credentials_path", type=str, default="../prod_env/credentials.conf")
parser.add_argument("--relative_cache_path", type=str, default="../prod_env/cache")
args = parser.parse_args()


print("========== Model infos ==========")
print(f"model_name: {args.model_name}")
print(f"tokenizer_name: {args.tokenizer_name}")
print(f"prefix: {args.prefix}")
print(f"suffix: {args.suffix}")
print("=================================")
print("")


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


credentials_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.relative_credentials_path)
credentials = get_credentials(credentials_path)
print("========== Credentials ==========")
for key, value in credentials.items():
    print(f"{key}: {value}")
print("=================================")
print("")

print("=========== Initial setup ===========")
current_path = os.path.dirname(os.path.realpath(__file__))
cache_path = os.path.join(current_path, args.relative_cache_path)
print(f"cache_path: {cache_path}")

client = AutoClient(credentials=credentials, cache_path=cache_path)
print("client successfully created")

print("Making short request...")
request = Request(model=args.model_name, prompt=args.prefix + "hello" + args.suffix, max_tokens=1)
response = client.make_request(request)
if not response.success:
    raise ValueError("Request failed")
print("Request successful")
print("=====================================")
print("")


def get_client():
    current_path = os.path.dirname(os.path.realpath(__file__))
    cache_path = os.path.join(current_path, args.relative_cache_path)
    client = AutoClient(credentials=credentials, cache_path=cache_path)
    return client


def get_clients(tokenizer_name: str):  # -> Tuple[Client, Client]:
    client = get_client()
    client_tokenizer = client._get_tokenizer_client(tokenizer_name)
    return client, client_tokenizer


def get_number_of_tokens(prompt: str, tokenizer_client: Any, tokenizer_name: str) -> int:
    tokenization_request = TokenizationRequest(tokenizer=tokenizer_name, text=prompt, encode=True)
    tokenization_response = tokenizer_client.tokenize(tokenization_request)
    return len(tokenization_response.tokens)


def try_request(
    client: Any,
    model_name: str,
    tokenizer_name: str,
    tokenizer_client: Any,
    sequence_length: int,
    num_tokens: int,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> bool:
    """
    Try to make a request with the given sequence_length and num_tokens.
    Return True if the request was successful, False otherwise.
    """
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    nb_tokens_prefix = get_number_of_tokens(prefix, tokenizer_client, tokenizer_name)
    nb_tokens_suffix = get_number_of_tokens(suffix, tokenizer_client, tokenizer_name)

    try:
        request = Request(
            model=model_name,
            prompt=prefix + " ".join(["hello"] * (sequence_length - nb_tokens_prefix - nb_tokens_suffix)) + suffix,
            max_tokens=num_tokens,
        )
        response = client.make_request(request)
        return response.success
    except Exception:
        return False


def figure_out_max_prompt_length(
    model_name: str,
    tokenizer_name: str,
    upper_bound: int = 9500,
    lower_bound: int = 450,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> Dict[str, int]:
    client, tokenizer_client = get_clients(tokenizer_name)
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    nb_tokens_prefix = get_number_of_tokens(prefix, tokenizer_client, tokenizer_name)
    nb_tokens_suffix = get_number_of_tokens(suffix, tokenizer_client, tokenizer_name)

    # Perform a dichotomy search to find the max tokens betzeen lower_bound and upper_bound
    lower_bound += nb_tokens_prefix + nb_tokens_suffix
    with tqdm(total=int(math.log2(upper_bound - lower_bound))) as pbar:
        while lower_bound < upper_bound:
            middle = math.ceil((lower_bound + upper_bound) / 2)
            if try_request(client, model_name, tokenizer_name, tokenizer_client, middle, 1, prefix, suffix):
                lower_bound = middle
            else:
                upper_bound = middle - 1
            pbar.update(1)

    # Just in case the number of tokens does not match the number of words, check number fo tokens with tokenizer
    max_prompt_length = get_number_of_tokens(
        prefix + " ".join(["hello"] * (lower_bound - nb_tokens_prefix - nb_tokens_suffix)) + suffix,
        tokenizer_client,
        tokenizer_name,
    )
    return {
        "max_prompt_length": max_prompt_length,
        "nb_tokens_prefix": nb_tokens_prefix,
        "nb_tokens_suffix": nb_tokens_suffix,
        "usable_max_prompt_length": max_prompt_length - nb_tokens_prefix - nb_tokens_suffix,
    }


def figure_out_max_prompt_length_plus_tokens(
    model_name: str,
    tokenizer_name: str,
    max_prompt_length: int,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> int:
    client, tokenizer_client = get_clients(tokenizer_name)
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    lower_bound = 1
    upper_bound = 2 * max_prompt_length + 1

    # Check if there is a limit (some model accept as many tokens as you want)
    if try_request(
        client,
        model_name,
        tokenizer_name,
        tokenizer_client,
        max_prompt_length,
        2**31 - 2 - max_prompt_length,
        prefix,
        suffix,
    ):
        print("The model has no limit on the number of tokens")
        return -1
    else:
        print("The model has a limit on the number of tokens")

    # Perform a dichotomy search to find the max tokens betzeen lower_bound and upper_bound
    with tqdm(total=int(math.log2(upper_bound - lower_bound))) as pbar:
        while lower_bound < upper_bound:
            middle = math.ceil((lower_bound + upper_bound) / 2)
            if try_request(
                client, model_name, tokenizer_name, tokenizer_client, max_prompt_length, middle, prefix, suffix
            ):
                lower_bound = middle
            else:
                upper_bound = middle - 1
            pbar.update(1)

    return lower_bound + max_prompt_length


def check_limits(
    model_name: str,
    tokenizer_name: str,
    infos: Dict[str, int],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> bool:
    client, tokenizer_client = get_clients(tokenizer_name)
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    result: bool = True

    # Check the max_prompt_length
    if "max_prompt_length" not in infos:
        raise ValueError("infos should contain max_prompt_length")
    max_prompt_length = infos["max_prompt_length"]
    if max_prompt_length < 0:
        print("No limit on the number of tokens")
        if not try_request(client, model_name, tokenizer_name, tokenizer_client, 2**32 - 2, 1, prefix, suffix):
            print(f"There is a limit on the number of tokens. Params: max_prompt_length={2**32 - 2}, max_tokens=1")
            result = False
    elif not try_request(client, model_name, tokenizer_name, tokenizer_client, max_prompt_length, 1, prefix, suffix):
        print(f"max_prompt_length is too big. Params: max_prompt_length={max_prompt_length}, max_tokens=1")
        result = False
    elif try_request(client, model_name, tokenizer_name, tokenizer_client, max_prompt_length + 1, 1, prefix, suffix):
        print(f"max_prompt_length could be bigger. Params: max_prompt_length={max_prompt_length+1}, max_tokens=1")
        result = False

    # Check the max_prompt_length_plus_tokens
    if "max_prompt_length_plus_tokens" not in infos:
        raise ValueError("infos should contain max_prompt_length_plus_tokens")
    max_prompt_length_plus_tokens = infos["max_prompt_length_plus_tokens"]
    # Generate r a raqndom number between 1 and max_prompt_length - 1
    r = random.randint(1, max_prompt_length - 1)
    if max_prompt_length_plus_tokens < 0:
        print("No limit on the number of tokens")
        if not try_request(
            client, model_name, tokenizer_name, tokenizer_client, max(1, max_prompt_length), 2**32 - 2, prefix, suffix
        ):
            print(
                f"There is a limit on the number of tokens. Params: max_prompt_length={max_prompt_length},"
                f" max_tokens={2**32 - 2}"
            )
            result = False
    elif not try_request(
        client,
        model_name,
        tokenizer_name,
        tokenizer_client,
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
    elif try_request(
        client,
        model_name,
        tokenizer_name,
        tokenizer_client,
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
    elif not try_request(
        client,
        model_name,
        tokenizer_name,
        tokenizer_client,
        max_prompt_length - r,
        max_prompt_length_plus_tokens - max_prompt_length + r,
        prefix,
        suffix,
    ):
        print(
            f"max_prompt_length_plus_tokens is too big. Params: max_prompt_length={max_prompt_length-r},"
            f" max_tokens={max_prompt_length_plus_tokens-max_prompt_length+r}"
        )
        result = False
    elif try_request(
        client,
        model_name,
        tokenizer_name,
        tokenizer_client,
        max_prompt_length - r,
        max_prompt_length_plus_tokens - max_prompt_length + r + 1,
        prefix,
        suffix,
    ):
        print(
            f"max_prompt_length_plus_tokens could be bigger. Params: max_prompt_length={max_prompt_length-r},"
            f" max_tokens={max_prompt_length_plus_tokens-max_prompt_length+r+1}"
        )
        result = False

    return result


print("========== Figure out max_prompt_length ==========")
infos = figure_out_max_prompt_length(args.model_name, args.tokenizer_name, prefix=args.prefix, suffix=args.suffix)
print(f"max_prompt_length: {infos['max_prompt_length']}")
print("===================================================")
print("")

print("========== Figure out max_prompt_length_plus_tokens ==========")
max_prompt_length_plus_tokens = figure_out_max_prompt_length_plus_tokens(
    args.model_name,
    args.tokenizer_name,
    max_prompt_length=infos["max_prompt_length"],
    prefix=args.prefix,
    suffix=args.suffix,
)
infos["max_prompt_length_plus_tokens"] = max_prompt_length_plus_tokens
print(f"max_prompt_length_plus_tokens: {infos['max_prompt_length_plus_tokens']}")
print("==============================================================")
print("")

# Check the limits
print("========== Check the limits ==========")
result = check_limits(args.model_name, args.tokenizer_name, infos, prefix=args.prefix, suffix=args.suffix)
if result:
    print("All limits are respected")
else:
    print("Some limits are not respected")
print("======================================")
print("")

# Print the infos
print("========== Print the infos ==========")
for key in infos:
    print(f"{key}: {infos[key]}")
print("=====================================")
