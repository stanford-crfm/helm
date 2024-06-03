"""Reads all runs from the suite and writes them to the CSV folder in CSV format.

EXPERIMENTAL: Not for public use.
TEMPORARY: Delete after 2024-09-30"""

import argparse
import csv
import os
import re

from tqdm import tqdm

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.codec import from_json
from helm.common.general import ensure_directory_exists


class FieldNames:
    CATEGORY_ID = "cate-idx"
    L2_NAME = "l2-name"
    L3_NAME = "l3-name"
    L4_NAME = "l4-name"
    PROMPT = "prompt"
    RESPONSE = "response"
    JUDGE_PROMPT = "judge_prompt"
    SCORE_REASON = "score_reason"
    SCORE = "score"


def process_one(scenario_state_path: str, csv_file_path: str):
    with open(scenario_state_path) as f:
        scenario_state = from_json(f.read(), ScenarioState)

    fieldnames = [
        FieldNames.CATEGORY_ID,
        FieldNames.L2_NAME,
        FieldNames.L3_NAME,
        FieldNames.L4_NAME,
        FieldNames.PROMPT,
        FieldNames.RESPONSE,
        FieldNames.JUDGE_PROMPT,
        FieldNames.SCORE_REASON,
        FieldNames.SCORE,
    ]
    with open(csv_file_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for request_state in scenario_state.request_states:
            row = {}
            references = request_state.instance.references
            assert len(references) == 4
            row[FieldNames.CATEGORY_ID] = references[0].output.text
            row[FieldNames.L2_NAME] = references[1].output.text
            row[FieldNames.L3_NAME] = references[2].output.text
            row[FieldNames.L4_NAME] = references[3].output.text
            row[FieldNames.PROMPT] = request_state.request.prompt
            assert request_state.result
            assert len(request_state.result.completions) == 1
            row[FieldNames.RESPONSE] = request_state.result.completions[0].text
            assert request_state.annotations
            row[FieldNames.JUDGE_PROMPT] = request_state.annotations["air_bench_2024"]["prompt_text"]
            row[FieldNames.SCORE_REASON] = request_state.annotations["air_bench_2024"]["reasoning"]
            row[FieldNames.SCORE] = request_state.annotations["air_bench_2024"]["score"]
            writer.writerow(row)
    print(f"Wrote {csv_file_path}")


def process_all(suite_path: str, csv_path: str):
    ensure_directory_exists(csv_path)
    run_dir_names = sorted([p for p in os.listdir(suite_path) if p.startswith("air_bench_2024:")])
    for run_dir_name in tqdm(run_dir_names, disable=None):
        scenario_state_path = os.path.join(suite_path, run_dir_name, "scenario_state.json")
        if not os.path.isfile(scenario_state_path):
            continue
        model_name_match = re.search("model=([A-Za-z0-9_-]+)", run_dir_name)
        assert model_name_match
        model_name = model_name_match[1]
        csv_file_path = os.path.join(csv_path, f"{model_name}_result.csv")
        process_one(scenario_state_path, csv_file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where the benchmarking output lives",
        default="benchmark_output",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Name of the CSV folder.",
        default="csv_output",
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite.",
        required=True,
    )
    args = parser.parse_args()
    suite_path = os.path.join(args.output_path, "runs", args.suite)
    process_all(suite_path, args.csv_path)


if __name__ == "__main__":
    main()
