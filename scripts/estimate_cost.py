import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

"""
Given a run suite directory, outputs metrics needed to estimate the cost of running.

Usage:
    python3 scripts/estimate_cost.py benchmark_output/runs/<Name of the run suite>
"""


@dataclass
class ModelCost:
    total_num_prompt_tokens: int = 0

    total_max_num_completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_num_prompt_tokens + self.total_max_num_completion_tokens

    def add_prompt_tokens(self, num_tokens: int):
        self.total_num_prompt_tokens += num_tokens

    def add_num_completion_tokens(self, num_tokens: int):
        self.total_max_num_completion_tokens += num_tokens


class CostCalculator:
    def __init__(self, run_suite_path: str):
        self._run_suite_path: str = run_suite_path

    def aggregate(self) -> Dict[str, ModelCost]:
        """Sums up the estimated number of tokens."""
        models_to_costs: Dict[str, ModelCost] = defaultdict(ModelCost)

        for run_dir in os.listdir(self._run_suite_path):
            run_path: str = os.path.join(self._run_suite_path, run_dir)

            if not os.path.isdir(run_path):
                continue

            run_spec_path: str = os.path.join(run_path, "run_spec.json")
            if not os.path.isfile(run_spec_path):
                continue

            # Extract the model name
            with open(run_spec_path) as f:
                run_spec = json.load(f)
                model: str = run_spec["adapter_spec"]["model"]

            metrics_path: str = os.path.join(run_path, "stats.json")
            with open(metrics_path) as f:
                metrics = json.load(f)

                for metric in metrics:
                    cost: ModelCost = models_to_costs[model]
                    metric_name: str = metric["name"]["name"]
                    if metric_name == "num_prompt_tokens":
                        cost.add_prompt_tokens(metric["sum"])
                    elif metric_name == "max_num_completion_tokens":
                        cost.add_num_completion_tokens(metric["sum"])

        return models_to_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    args = parser.parse_args()

    calculator = CostCalculator(args.run_suite_path)
    model_costs: Dict[str, ModelCost] = calculator.aggregate()
    for model_name, model_cost in model_costs.items():
        print(
            f"{model_name}: Total prompt tokens={model_cost.total_num_prompt_tokens} + "
            f"Total max completion tokens={model_cost.total_max_num_completion_tokens} = "
            f"{model_cost.total_tokens}"
        )
