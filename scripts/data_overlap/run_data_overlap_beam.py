import json
import apache_beam as beam

from typing import Callable

from data_overlap_beam import ComputeAndWriteDataOverlapStats
from common.arguments import get_data_overlap_args


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
    args = get_data_overlap_args()

    extract_text_from_document: Callable[[str], str] = get_extract_text_function(args.input_format)

    # The model developer should pass in the appropriate PipelineOptions here.
    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            # The model developer should modify these lines to read from the actual training set.
            | "Read" >> beam.io.ReadFromText(args.input_data)
            | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
            # Call the HELM Overlap Apache Beam API.
            | "ComputeAndWriteDataOverlapStats"
            >> ComputeAndWriteDataOverlapStats(
                scenario_data_path=args.scenario_data,
                n_values=args.N,
                normalization=args.normalization,
                tags={"tags:": args.tags},
                output_stats=args.output_stats,
            )
        )
    print(f"Wrote results to {args.output_stats}")


if __name__ == "__main__":
    main()
