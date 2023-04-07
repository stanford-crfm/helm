import json
import apache_beam as beam

from typing import Callable, Dict, List

from helm.benchmark.contamination.contamination_parser import create_parser
from contamination_beam import ComputeAndWriteContaminationStats


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
    parser = create_parser()
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
