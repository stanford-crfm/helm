import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

"""
Given a run suite directory, outputs metrics needed to estimate the cost of running.

Usage:
    python3 scripts/estimate_cost.py benchmark_output/runs/<Name of the run suite>
"""


@dataclass
class ModelCost:
    total_num_prompt_tokens: int = 0

    total_num_completion_tokens: int = 0

    total_num_instances: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_num_prompt_tokens + self.total_num_completion_tokens

    def add_prompt_tokens(self, num_tokens: int):
        self.total_num_prompt_tokens += num_tokens

    def add_num_completion_tokens(self, num_tokens: int):
        self.total_num_completion_tokens += num_tokens

    def add_num_instances(self, num_instances: int):
        self.total_num_instances += num_instances


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
                cost: ModelCost = models_to_costs[model]

            metrics_path: str = os.path.join(run_path, "stats.json")
            with open(metrics_path) as f:
                metrics: List[Dict] = json.load(f)
                if len(metrics) == 0:
                    continue

                num_prompt_tokens: int = -1
                num_completion_tokens: int = -1
                num_instances: int = -1

                for metric in metrics:
                    metric_name: str = metric["name"]["name"]

                    # Don't count perturbations
                    if "perturbation" in metric["name"]:
                        continue

                    if metric_name == "num_prompt_tokens":
                        num_prompt_tokens = metric["sum"]
                    elif metric_name == "num_completion_tokens":
                        num_completion_tokens = metric["sum"]
                    elif metric_name == "num_instances":
                        num_instances = metric["sum"]

                assert (
                    num_prompt_tokens >= 0 and num_completion_tokens >= 0 and num_instances >= 0
                ), f"invalid metrics: {metrics}"
                cost.add_prompt_tokens(num_prompt_tokens * num_instances)
                cost.add_num_completion_tokens(num_completion_tokens * num_instances)
                cost.add_num_instances(num_instances)

        return models_to_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    args = parser.parse_args()

    calculator = CostCalculator(args.run_suite_path)
    model_costs: Dict[str, ModelCost] = calculator.aggregate()

    grand_total_prompt_tokens: int = 0
    grand_total_completion_tokens: int = 0
    grand_total_num_instances: int = 0
    for model_name, model_cost in model_costs.items():
        print(
            f"{model_name}: Total prompt tokens={model_cost.total_num_prompt_tokens} + "
            f"Total completion tokens={model_cost.total_num_completion_tokens} = "
            f"{model_cost.total_tokens} for {model_cost.total_num_instances} instances."
        )

        grand_total_prompt_tokens += model_cost.total_num_prompt_tokens
        grand_total_completion_tokens += model_cost.total_num_completion_tokens
        grand_total_num_instances += model_cost.total_num_instances

    print(
        f"Grand total prompt tokens={grand_total_prompt_tokens} + "
        f"Grand total completion tokens={grand_total_completion_tokens} = "
        f"{grand_total_prompt_tokens + grand_total_completion_tokens} for {grand_total_num_instances} instances."
    )
