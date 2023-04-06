import argparse


def create_parser():
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
        "--output-ngrams",
        type=str,
        default=None,
        help="Path to the file of contaminated ngrams. To output the ngrams, you must also specify --max-output-ngrams",
    )
    parser.add_argument(
        "--max-output-ngrams",
        type=int,
        default=0,
        help=(
            "The max number of contaminated ngrams to be stored for each (n, light_instance, part)."
            "Set to -1 to store all"
        ),
    )
    return parser
