import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


"""
Given a run suite directory, calculates how many GooseAI credits (in dollars) is needed.

Usage:
    python3 scripts/calculate_gooseai_credits.py benchmark_output/runs/<Name of the run suite>
"""


@dataclass
class ChargeableEntities:
    number_of_requests: int = 0
    number_of_tokens: int = 0


class GooseAICreditsCalculator:
    # Source: https://goose.ai/pricing
    GPT_J_PRICE_PER_REQUEST: float = 0.000450
    GPT_J_PRICE_PER_TOKEN: float = 0.000012

    GPT_NEO_PRICE_PER_REQUEST: float = 0.002650
    GPT_NEO_PRICE_PER_TOKEN: float = 0.000063

    @staticmethod
    def compute_credits(models_to_chargeable_entities: Dict[str, ChargeableEntities]) -> float:
        """Compute credits based on https://goose.ai/pricing."""
        total: float = 0
        for model, chargeable_entities in models_to_chargeable_entities.items():
            tokens: int = chargeable_entities.number_of_tokens
            requests: int = chargeable_entities.number_of_requests

            print(f"{model}: {tokens:,} tokens, {requests:,} requests")
            if model == "gooseai/gpt-j-6b":
                total += tokens * GooseAICreditsCalculator.GPT_J_PRICE_PER_TOKEN
                total += requests * GooseAICreditsCalculator.GPT_J_PRICE_PER_REQUEST
            elif model == "gooseai/gpt-neo-20b":
                total += tokens * GooseAICreditsCalculator.GPT_NEO_PRICE_PER_TOKEN
                total += requests * GooseAICreditsCalculator.GPT_NEO_PRICE_PER_REQUEST
            else:
                raise ValueError(f"Unknown model: {model}")

        return total

    def __init__(self, run_suite_path: str):
        self.run_suite_path: str = run_suite_path

    def calculate(self) -> float:
        """Sums up estimated token cost and converts it to dollars."""
        models_to_chargeable_entities: Dict[str, ChargeableEntities] = defaultdict(ChargeableEntities)

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

                print(f"{run_dir.replace(',', '_')}, tokens={metrics[0]['sum']}, requests={metrics[1]['sum']}")
                for metric in metrics:
                    if metric["name"]["name"] == "estimated_number_of_tokens":
                        models_to_chargeable_entities[model].number_of_tokens += metric["sum"]
                    elif metric["name"]["name"] == "number_of_requests":
                        models_to_chargeable_entities[model].number_of_requests += metric["sum"]

        return GooseAICreditsCalculator.compute_credits(models_to_chargeable_entities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    args = parser.parse_args()

    calculator = GooseAICreditsCalculator(args.run_suite_path)
    total_credits: float = calculator.calculate()
    print(f"\nTotal credits: ${total_credits:.2f}")
