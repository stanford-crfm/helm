import argparse
import json
from typing import Any

def save_metrics_to_jsonl(overlap_metrics, filename: str):
    with open(filename, "w") as f:
        for overlap_metric in overlap_metrics:
            f.write(json.dumps(overlap_metric, ensure_ascii=False) + "\n")

def update_metrics(metrics_path, out_path):
    overlap_metrics_jsons = open(metrics_path, "r").readlines()

    entry_overlap_metric_list = []
    for entry_overlap_metric_json in overlap_metrics_jsons:
        entry_overlap_metric_dict = json.loads(entry_overlap_metric_json)
        entry_overlap_metric_dict['entry_data_overlap_key']['instance_id'] = ''
        entry_overlap_metric_list.append(entry_overlap_metric_dict)

    save_metrics_to_jsonl(entry_overlap_metric_list, out_path)


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=str, required=True, help="Path to your metrics")
    parser.add_argument("--out-path", type=str, required=True, help="Path to the output metrics file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    update_metrics(args.metrics_path, args.out_path)