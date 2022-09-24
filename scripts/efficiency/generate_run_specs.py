import argparse
from datetime import datetime


def generate_spec(scenario, model, tokenizer, num_prompt_tokens, num_output_tokens, random):
    return (
        f'"{scenario}:model={model},tokenizer={tokenizer},'
        f"num_prompt_tokens={num_prompt_tokens},"
        f"num_output_tokens={num_output_tokens},"
        f'random={random}": '
        '{status: "READY", priority: 1}'
    )


def main(args):
    now = datetime.now()
    current_date = now.strftime("%m/%d/%Y")

    all_num_prompt_tokens = [1, 256, 512, 1024, 1536]
    all_num_output_tokens = [1, 2, 4, 8, 16, 32, 64]

    scenario = "synthetic_efficiency"
    if args.model == "ai21":
        all_models_and_tokenizers = [("ai21_tokenizer", "ai21/j1")]
    elif args.model == "anthropic":
        all_models_and_tokenizers = [("anthropic/stanford-online-all-v4-s3", "huggingface/gpt2")]
    elif args.model == "openai":
        all_models_and_tokenizers = [("gpt2_tokenizer", "huggingface/gpt2")]
    elif args.model == "cohere":
        all_models_and_tokenizers = [("cohere_tokenizer", "cohere/cohere")]
    elif args.model == "opt":
        all_models_and_tokenizers = [("opt_tokenizer", "facebook/opt-66b")]

    specs = []

    num_specs = len(all_models_and_tokenizers) * len(all_num_prompt_tokens) * len(all_num_output_tokens)
    print(f"Generating {num_specs} specs...")

    for model, tokenizer in all_models_and_tokenizers:
        for num_prompt_tokens in all_num_prompt_tokens:
            for num_output_tokens in all_num_output_tokens:
                spec = generate_spec(
                    scenario, model, tokenizer, num_prompt_tokens, num_output_tokens, random=current_date
                )
                specs.append(spec)

    print(f"Writing out {len(specs)} specs...")
    with open(args.output_path, "w") as f:
        for spec in specs:
            f.write(f"{spec}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ai21", "anthropic", "openai", "cohere", "opt"], required=True)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()
    main(args)
