import argparse
import json
import os
from typing import List, Tuple

"""
Extracts all the prompts from a suite and writes it out to a TSV file,
making it easier to double check the constructed prompts look reasonable.

Usage:
     python3 scripts/extract_prompts.py benchmark_output/runs/<Name of the run suite>
"""


class PromptsExtractor:
    def __init__(self, run_suite_path: str):
        self.run_suite_path: str = run_suite_path

    def extract(self, output_path: str) -> None:
        prompts: List[Tuple[str, str]] = []

        for run_dir in os.listdir(self.run_suite_path):
            run_path: str = os.path.join(self.run_suite_path, run_dir)

            if not os.path.isdir(run_path):
                continue

            scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
            if not os.path.isfile(scenario_state_path):
                continue

            with open(scenario_state_path, "r") as f:
                scenario_state = json.load(f)
                for request_state in scenario_state["request_states"]:
                    prompt: str = request_state["request"]["prompt"]
                    prompts.append((prompt, run_dir))

        with open(output_path, "w") as f:
            f.write("Prompts\tRunSpecs\n")
            for prompt, run_dir in prompts:
                f.write(f"{prompt}\t{run_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_suite_path", type=str, help="Path to runs folder")
    parser.add_argument("--output-path", type=str, default="prompts.tsv")
    args = parser.parse_args()

    PromptsExtractor(args.run_suite_path).extract(args.output_path)
    print("Done.")
