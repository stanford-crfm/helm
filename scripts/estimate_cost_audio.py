import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

"""
Given a run suite directory, computes and aggregates metrics needed to estimate the cost of running.

Usage:
    python3 scripts/estimate_cost_audio.py benchmark_output/runs/<Name of the run suite>
"""


@dataclass
class ModelCost:
    total_num_prompt_chars: int = 0
    total_num_completion_chars: int = 0
    total_num_audio_files: int = 0
    total_num_instances: int = 0

    def add_prompt_chars(self, num_chars: int):
        self.total_num_prompt_chars += num_chars

    def add_completion_chars(self, num_chars: int):
        self.total_num_completion_chars += num_chars

    def add_audio_files(self, num_files: int):
        self.total_num_audio_files += num_files

    def add_num_instances(self, num_instances: int):
        self.total_num_instances += num_instances


class CostCalculator:
    def __init__(self, run_suite_path: str):
        self._run_suite_path: str = run_suite_path

    def aggregate(self) -> Dict[str, ModelCost]:
        """Sums up the estimated number of characters and audio files."""
        models_to_costs: Dict[str, ModelCost] = defaultdict(ModelCost)

        for run_dir in os.listdir(self._run_suite_path):
            run_path: str = os.path.join(self._run_suite_path, run_dir)

            if not os.path.isdir(run_path):
                continue

            scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
            if not os.path.isfile(scenario_state_path):
                continue

            with open(scenario_state_path) as f:
                data = json.load(f)
                model: str = data["adapter_spec"]["model"]
                cost: ModelCost = models_to_costs[model]

                for request_state in data.get("request_states", []):
                    request = request_state.get("request", {})
                    result = request_state.get("result", {})
                    completions = result.get("completions", [])

                    multimodal_prompt = request.get("multimodal_prompt", {})
                    media_objects = multimodal_prompt.get("media_objects", [])

                    prompt_chars = sum(
                        len(obj.get("text", "")) for obj in media_objects if obj.get("content_type") == "text/plain"
                    )

                    num_audio_files = sum(
                        1 for obj in media_objects if obj.get("content_type", "").startswith("audio/")
                    )

                    completion_chars = sum(len(comp.get("text", "")) for comp in completions)

                    cost.add_prompt_chars(prompt_chars)
                    cost.add_completion_chars(completion_chars)
                    cost.add_audio_files(num_audio_files)
                    cost.add_num_instances(1)

        return models_to_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    args = parser.parse_args()

    calculator = CostCalculator(args.run_suite_path)
    model_costs: Dict[str, ModelCost] = calculator.aggregate()

    grand_total_prompt_chars: int = 0
    grand_total_completion_chars: int = 0
    grand_total_audio_files: int = 0
    grand_total_num_instances: int = 0

    for model_name, model_cost in model_costs.items():
        print(
            f"{model_name}: Prompt chars={model_cost.total_num_prompt_chars}, "
            f"Completion chars={model_cost.total_num_completion_chars}, "
            f"Audio files={model_cost.total_num_audio_files}, "
            f"Instances={model_cost.total_num_instances}"
        )

        grand_total_prompt_chars += model_cost.total_num_prompt_chars
        grand_total_completion_chars += model_cost.total_num_completion_chars
        grand_total_audio_files += model_cost.total_num_audio_files
        grand_total_num_instances += model_cost.total_num_instances

    print(
        f"\nGRAND TOTALS:\n"
        f"Prompt chars: {grand_total_prompt_chars}\n"
        f"Completion chars: {grand_total_completion_chars}\n"
        f"Audio files: {grand_total_audio_files}\n"
        f"Instances: {grand_total_num_instances}"
    )
