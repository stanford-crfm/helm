import json
import apache_beam as beam
import argparse

from typing import Callable, List

from scripts.data_contamination.contamination_beam import ComputeAndWriteContaminationStats


def get_extract_text_function(input_format: str):
    def extract_text_from_the_pile_document(document: str) -> str:
        return json.loads(document)["text"]

    def extract_text_from_raw_document(document: str) -> str:
        return document.rstrip("\n")

    if input_format == "raw":
        return extract_text_from_raw_document
    elif input_format == "the_pile":
        return extract_text_from_the_pile_document
    else:
        raise NotImplementedError(f"Unknown input format {input_format}")


def main():
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
    args = parser.parse_args()

    n_values: List[int] = [5, 9, 13]  # TODO: Pick the N values
    extract_text_from_document: Callable[[str], str] = get_extract_text_function(args.input_format)

    # The model developer should pass in the appropriate PipelineOptions here.
    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            # The model developer should modify these lines to read from the actual training set.
            | "Read" >> beam.io.ReadFromText(args.input_data)
            | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
            # Call the HELM Contamination Apache Beam API.
            | "ComputeAndWriteContaminationStats"
            >> ComputeAndWriteContaminationStats(
                scenario_data_path=args.scenario_data,
                n_values=n_values,
                normalization=args.normalization,
                tags={"tags:": args.tags},
                output_stats=args.output_stats,
            )
        )
    print(f"Wrote results to {args.output_stats}")


if __name__ == "__main__":
    main()
