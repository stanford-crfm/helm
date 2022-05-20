import argparse
import json
import os
from collections import defaultdict
from typing import Dict

"""
Given a run suite directory, calculates how much OpenAI credits we will need.

Usage:

    python3 scripts/calculate_openai_credits.py benchmark_output/runs/<Name of the run suite>

"""


class OpenAICreditsCalculator:
    # Source: https://openai.com/api/pricing/
    ADA_PRICE_PER_TOKEN: float = 0.0008 / 1000
    BABBAGE_PRICE_PER_TOKEN: float = 0.0012 / 1000
    CURIE_PRICE_PER_TOKEN: float = 0.0060 / 1000
    DAVINCI_PRICE_PER_TOKEN: float = 0.0600 / 1000

    @staticmethod
    def compute_credits(models_to_tokens: Dict[str, float]) -> float:
        """
        Given the number of tokens we need for each model,
        compute total credits based on https://openai.com/api/pricing.
        """
        total: float = 0
        for model, tokens in models_to_tokens.items():
            if "openai" not in model:
                continue

            print(f"{model}: {tokens} tokens total")
            if "ada" in model:
                total += tokens * OpenAICreditsCalculator.ADA_PRICE_PER_TOKEN
            elif "babbage" in model:
                total += tokens * OpenAICreditsCalculator.BABBAGE_PRICE_PER_TOKEN
            elif "curie" in model:
                total += tokens * OpenAICreditsCalculator.CURIE_PRICE_PER_TOKEN
            elif "davinci" in model or "code" in model:
                # TODO: Not sure what the price is for codex (still in beta), but we don't use these models much
                total += tokens * OpenAICreditsCalculator.DAVINCI_PRICE_PER_TOKEN
            else:
                raise ValueError(f"Unknown model: {model}")

        return total

    def __init__(self, run_suite_path: str):
        self.run_suite_path: str = run_suite_path

    def calculate(self) -> float:
        """
        Sums up the estimated number of tokens we need and converts it to dollars
        using OpenAI's pricing model.
        """
        models_to_tokens: Dict[str, float] = defaultdict(float)

        for run_dir in os.listdir(self.run_suite_path):
            run_path: str = os.path.join(self.run_suite_path, run_dir)

            if not os.path.isdir(run_path):
                continue

            run_spec_path: str = os.path.join(run_path, "run_spec.json")
            if not os.path.isfile(run_spec_path):
                continue

            with open(run_spec_path) as f:
                run_spec = json.load(f)
                model: str = run_spec["adapter_spec"]["model"]

            metrics_path: str = os.path.join(run_path, "metrics.json")
            with open(metrics_path) as f:
                metrics = json.load(f)

                for metric in metrics:
                    if metric["name"]["name"] != "estimated_number_of_tokens":
                        continue

                    models_to_tokens[model] += metrics[0]["sum"]

        return OpenAICreditsCalculator.compute_credits(models_to_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    args = parser.parse_args()

    calculator = OpenAICreditsCalculator(args.run_suite_path)
    total_credits: float = calculator.calculate()
    print(f"\nTotal credits: ${total_credits:.2f}")
