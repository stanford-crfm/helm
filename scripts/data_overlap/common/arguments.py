import argparse
from typing import Any


def get_data_overlap_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="Path to your training data")
    parser.add_argument("--scenario-data", type=str, required=True, help="Path to scenario data (benchmarking data)")
    parser.add_argument("--output-stats", type=str, required=True, help="Path to the output file")
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        help="The format of your input file for your training data, e.g. raw, custom, the_pile",
    )
    parser.add_argument("--no-output-ngrams", type=bool, default=False, help="Pass to not output ngrams")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Other tags, such as whether the input data is for pretraining or instruction tuning",
    )
    parser.add_argument(
        "--normalization", type=str, default="default", help="What normalization and tokenization strategy to apply"
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="*",
        default=[5, 9, 13],
        help="N for ngrams that we want to run, defaults to 5, 9, 13",
    )
    return parser.parse_args()
