import os
from typing import Dict, List

from common.general import ensure_directory_exists, ensure_file_downloaded, parse_hocon, write
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from proxy.auto_client import AutoClient
from proxy.service import (
    CREDENTIALS_FILE,
    CACHE_DIR,
)
from benchmark.synthetic_efficiency_scenario import NUM_INPUT_TOKENS

MAX_ITERS = 5


def _count_prompt_tokens(client: AutoClient, prompt: str, tokenizer: str):
    request: TokenizationRequest = TokenizationRequest(text=prompt, tokenizer=tokenizer)
    result: TokenizationRequestResult = client.tokenize(request)
    return len(result.tokens)


def generate_synthetic_efficiency_instances(
    num_instances: int,
    num_input_tokens: int,
    tokenizer: str,
    output_path: str = "synthetic_efficiency_instances",
    base_path: str = "prod_env",
):
    """Generates prompts for the synthetic efficiency scenario."""
    sources = {
        "alice": "https://www.gutenberg.org/files/11/11-0.txt",
        "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "monte_cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
        "crime_and_punishment": "https://www.gutenberg.org/files/2554/2554-0.txt",
    }

    credentials_path = os.path.join(base_path, CREDENTIALS_FILE)
    cache_path = os.path.join(base_path, CACHE_DIR)
    ensure_directory_exists(cache_path)
    if os.path.exists(credentials_path):
        with open(credentials_path) as f:
            credentials = parse_hocon(f.read())
    else:
        credentials = {}

    client = AutoClient(credentials, cache_path)

    books: List[str] = list(sources.keys())
    tokens: Dict = {}
    text_chunks: Dict = {}

    tokenizer_organization: str = tokenizer.split("/")[0]
    huggingface_tokenizer: bool = tokenizer_organization == "huggingface"

    # Extract tokens from book sources
    seen_tokens = set()
    for book, source_url in sources.items():
        data_path = os.path.join(output_path, f"{book}.txt")
        ensure_file_downloaded(
            source_url=source_url, target_path=data_path, unpack=False,
        )
        with open(data_path, "r") as f:
            raw_text = f.read()
        batch_size = 2048
        # Skip intro text
        text = raw_text.split(" ")[batch_size:]
        num_total_tokens_per_book = (num_instances * num_input_tokens) // len(books)
        i = 0
        tokens[book] = []
        text_chunks[book] = []
        while len(tokens[book]) < num_total_tokens_per_book:
            batch = " ".join(text[i * batch_size : (i + 1) * batch_size])
            while True:
                request: TokenizationRequest = TokenizationRequest(
                    text=batch, tokenizer=tokenizer, encode=huggingface_tokenizer
                )
                result: TokenizationRequestResult = client.tokenize(request)
                tokens_ = frozenset([token.value for token in result.tokens])
                if tokens_ not in seen_tokens:
                    seen_tokens.add(tokens_)
                    break
            tokens[book] += result.tokens
            if not huggingface_tokenizer:
                text_chunks[book] += [
                    result.text[token.text_range.start : token.text_range.end] for token in result.tokens
                ]
            i += 1

    prompts = []
    for i in range(num_instances // len(books)):
        for j in range(len(books)):
            prompt: str = ""

            # Initialize
            if huggingface_tokenizer:
                per_instance_tokens = [
                    token.value for token in tokens[books[j]][i * num_input_tokens : (i + 1) * num_input_tokens]
                ]
            else:
                per_instance_tokens = text_chunks[books[j]][i * num_input_tokens : (i + 1) * num_input_tokens]

            # Iterate until we get the right number of tokens
            success = False
            num_iters = 0
            while num_iters < MAX_ITERS:
                if huggingface_tokenizer:
                    decode_request: DecodeRequest = DecodeRequest(tokens=per_instance_tokens)
                    decode_result: DecodeRequestResult = client.decode(decode_request)
                    prompt = decode_result.text
                else:
                    prompt = "".join(per_instance_tokens)

                num_generated_tokens = _count_prompt_tokens(client, prompt, tokenizer)
                if num_generated_tokens != num_input_tokens:
                    temp_num_tokens = num_generated_tokens
                    while temp_num_tokens < num_input_tokens:
                        if len(per_instance_tokens) == 0:
                            assert num_input_tokens == 1
                            if huggingface_tokenizer:
                                per_instance_tokens = tokens[books[j]][:2]
                            else:
                                per_instance_tokens = text_chunks[books[j]][:2]
                        else:
                            per_instance_tokens.append(per_instance_tokens[-1])
                        temp_num_tokens += 1
                    while temp_num_tokens > num_input_tokens:
                        per_instance_tokens = per_instance_tokens[:-1]
                        temp_num_tokens -= 1
                else:
                    success = True
                    break
                num_iters += 1
            if not success:
                raise RuntimeError(
                    f"Requested {num_input_tokens}, got {num_generated_tokens} for "
                    f"book {books[j]}, instance #{i}, tokenizer={tokenizer}"
                )
            prompts.append(prompt)

    for i, prompt in enumerate(prompts):
        name = f"input_tokens={num_input_tokens}," f"tokenizer={tokenizer.replace('/', '_')}," f"id={i}.txt"
        write(os.path.join(output_path, name), prompt)


if __name__ == "__main__":
    for num_input_tokens in NUM_INPUT_TOKENS:
        for tokenizer in ["huggingface/gpt2_tokenizer_fast", "ai21"]:
            generate_synthetic_efficiency_instances(
                num_instances=10, num_input_tokens=num_input_tokens, tokenizer=tokenizer,
            )
