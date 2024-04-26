"""
This script generates run_specs for the synthetic_efficiency_scenario.
"""

import argparse
from datetime import datetime


def generate_spec(scenario, model, tokenizer, num_prompt_tokens, num_output_tokens, random):
    random_str: str = ""
    if random is not None:
        random_str = f",random={random}"
    return (
        f'"{scenario}:model={model},tokenizer={tokenizer},'
        f"num_prompt_tokens={num_prompt_tokens},"
        f"num_output_tokens={num_output_tokens}"
        f'{random_str}": '
        "{priority: 1}"
    )


def main(args):
    now = datetime.now()
    current_date = now.strftime("%m/%d/%Y")

    all_num_prompt_tokens = [1, 256, 512, 1024, 1536]
    all_num_output_tokens = [1, 2, 4, 8, 16, 32, 64]

    scenario = "synthetic_efficiency"
    all_models_and_tokenizers = []
    for tokenizer_provider in args.tokenizer_providers:
        if tokenizer_provider == "ai21":
            all_models_and_tokenizers.append(("ai21_tokenizer", "ai21/j1"))
        elif tokenizer_provider == "openai":
            all_models_and_tokenizers.append(("gpt2_tokenizer", "huggingface/gpt2"))
        elif tokenizer_provider == "cohere":
            all_models_and_tokenizers.append(("cohere_tokenizer", "cohere/cohere"))
        elif tokenizer_provider == "opt":
            all_models_and_tokenizers.append(("opt_tokenizer", "meta/opt"))
        elif tokenizer_provider == "yandex":
            all_models_and_tokenizers.append(("together/yalm", "yandex/yalm"))
        elif tokenizer_provider == "bloom":
            all_models_and_tokenizers.append(("together/bloom", "bigscience/bloom"))
        elif tokenizer_provider == "t5":
            all_models_and_tokenizers.append(("together/t5-11b", "google/t5"))
        elif tokenizer_provider == "t0":
            all_models_and_tokenizers.append(("together/t0pp", "bigscience/t0pp"))
        elif tokenizer_provider == "ul2":
            all_models_and_tokenizers.append(("together/ul2", "google/ul2"))
        elif tokenizer_provider == "glm":
            all_models_and_tokenizers.append(("together/glm", "tsinghua/glm"))
        elif tokenizer_provider == "gptj":
            all_models_and_tokenizers.append(("together/gpt-j-6b", "eleutherai/gptj"))
        elif tokenizer_provider == "gptneox":
            all_models_and_tokenizers.append(("together/gpt-neox-20b", "eleutherai/gptneox"))

    specs = []
    num_specs = len(all_models_and_tokenizers) * len(all_num_prompt_tokens) * len(all_num_output_tokens)
    print(f"Generating {num_specs} specs...")
    for model, tokenizer in all_models_and_tokenizers:
        for num_prompt_tokens in all_num_prompt_tokens:
            for num_output_tokens in all_num_output_tokens:
                if args.no_random:
                    random = None
                else:
                    random = current_date
                spec = generate_spec(scenario, model, tokenizer, num_prompt_tokens, num_output_tokens, random=random)
                specs.append(spec)

    print(f"Writing out {len(specs)} specs...")
    with open(args.output_path, "w") as f:
        for spec in specs:
            f.write(f"{spec}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_providers",
        choices=["ai21", "openai", "cohere", "opt", "yandex", "bloom", "t5", "t0", "ul2", "glm", "gptj", "gptneox"],
        required=True,
        nargs="+",
    )
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--no-random", action="store_true")
    args = parser.parse_args()
    main(args)
