import json
import argparse
import os
import glob

from typing import List
from nltk import ngrams
from typing import Dict, Optional, DefaultDict
from collections import defaultdict

from helm.benchmark.contamination.light_scenario import LightInstance, LightScenario
from helm.benchmark.contamination.light_tokenizer import LightTokenizer, DefaultTokenizer
from helm.benchmark.contamination.document_reading_processor import DocumentReadingProcessor
from helm.benchmark.contamination.contamination_stats import ContaminationStats, PART_INPUT, PART_REF


# The n values of the ngrams to be computed
N_VALUES: List[int] = [5, 9, 13]  # TODO: Pick the N values


def load_scenarios_from_jsonl(path: str) -> List[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file, where each json represents a LightScenario object.

    Input file format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...

    Each line is a json and each json looks like:
    {
        "scenario_spec": "SCENARIO_SPEC",
        "light_instances": [
            {
                "input": "INPUT_TEXT1",
                "references": [
                    "REFERENCE_TEXT_1",
                    "REFERENCE_TEXT_2"
                ]
            },
            {
                "input": "INPUT_TEXT2",
                "references": [
                    "REFERENCE_TEXT_3",
                    "REFERENCE_TEXT_4"
                ]
            }
        ]
    }
    """

    def create_light_instance_from_dict(instance_dict: dict) -> LightInstance:
        return LightInstance(input=instance_dict["input"], references=instance_dict["references"])

    light_scenarios: List[LightScenario] = []
    scenario_jsons = open(path, "r").readlines()
    for scenario_json in scenario_jsons:
        scenario_dict: dict = json.loads(scenario_json)
        scenario_spec: str = scenario_dict["scenario_spec"]
        light_instances: List[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in scenario_dict["light_instances"]
        ]
        light_scenarios.append(LightScenario(scenario_spec=scenario_spec, light_instances=light_instances))
    return light_scenarios


def compute_scenario_file_contamination(
    scenarios: List[LightScenario],
    training_file_path: str,
    n_values: List[int],
    file_format: str,
    ngram_index: DefaultDict[int, DefaultDict[str, List]],
    tokenizer: LightTokenizer,
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]] = None,
) -> Dict[str, ContaminationStats]:
    """Given an input file, compute a contamination stats for each n and each scenario"""

    all_scenario_stats: Dict[str, ContaminationStats] = {}
    for scenario in scenarios:
        for n in n_values:
            # Initizlize a stats instance for every pair of <scenario, n>
            stats: ContaminationStats = ContaminationStats.from_scenario(scenario, stats_tags=[f"N={n}"])
            all_scenario_stats[stats.stats_repr] = stats

    document_generator = DocumentReadingProcessor(
        file_path=training_file_path, file_format=file_format
    ).get_document_generator()
    document_index: int = 0
    for document in document_generator:
        document_index += 1
        print(f"Processing the {document_index}th document...", end="\r")
        compute_scenario_document_contamination(
            ngram_index=ngram_index,
            document=document,
            n_values=n_values,
            tokenizer=tokenizer,
            overlapped_ngrams=overlapped_ngrams,
        )

    return all_scenario_stats


def compute_scenario_document_contamination(
    ngram_index: Dict[int, DefaultDict[str, list]],
    document: str,
    n_values: List[int],
    tokenizer: LightTokenizer,
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]] = None,
):
    """Given a document, compute a contamination stats for each n and each scenario"""
    document_unigrams = tokenizer.tokenize(document)
    for n in n_values:
        assert n in ngram_index
        for document_ngram in ngrams(document_unigrams, n):
            if document_ngram in ngram_index[n]:
                if overlapped_ngrams is not None:
                    overlapped_ngrams[n][str(document_ngram)] += 1
                for contamination_stats, instance_id, part in ngram_index[n][document_ngram]:
                    contamination_stats.write_dirty(instance_id, part)


if __name__ == "__main__":
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
        nargs="+",
        help="Other tags, such as whether the input data is for pretraining or instruction tuning",
        default=[],
    )
    parser.add_argument(
        "--normalization", type=str, default="default", help="What normalization and tokenization strategy to apply"
    )
    parser.add_argument(
        "--output-ngrams",
        type=str,
        default=None,
        help="Path to the file of overlapped ngrams. If not given, ngrams will not be output.",
    )

    args = parser.parse_args()

    tokenizer: LightTokenizer
    if args.normalization == "none":
        tokenizer = LightTokenizer()
    elif args.normalization == "default":
        tokenizer = DefaultTokenizer()
    else:
        raise ValueError(f"Normalization strategy {args.normalization} is not defined.")

    input_file_paths: List[str]
    if os.path.isdir(args.input_data):
        input_file_paths = []
        for file_path in glob.iglob(os.path.join(args.input_data, "**/*"), recursive=True):
            if os.path.isfile(file_path):
                input_file_paths.append(file_path)
    else:
        input_file_paths = [args.input_data]
    print(f"The input data will be loaded from {input_file_paths}")

    print(f"Loading scenario data from {args.scenario_data}")
    scenarios = load_scenarios_from_jsonl(args.scenario_data)

    # initialize the stats and ngram_index
    all_contamination_stats: Dict[str, ContaminationStats] = {}
    ngram_index: DefaultDict[int, DefaultDict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    for scenario in scenarios:
        print(f"Building ngram indexes for {scenario.scenario_spec}")
        for n in N_VALUES:
            # Initizlize a stats instance for every pair of <scenario, n>
            stats: ContaminationStats = ContaminationStats.from_scenario(scenario, stats_tags=[f"N={n}"])
            if stats.stats_repr in all_contamination_stats:
                raise ValueError("Duplicated settings detected.")
            else:
                all_contamination_stats[stats.stats_repr] = stats

            # Build the ngram index
            for i in range(len(scenario.light_instances)):
                instance = scenario.light_instances[i]
                # compute input ngrams
                # TODO: The unigram computation can be taken out to further optimize efficiency if necessary.
                input_unigrams = tokenizer.tokenize(instance.input)
                for input_ngram in ngrams(input_unigrams, n):
                    ngram_index[n][input_ngram].append((stats, i, PART_INPUT))

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        ngram_index[n][reference_ngram].append((stats, i, PART_REF))

    # Initialize overlapped_ngrams
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]]
    if args.output_ngrams is None:
        overlapped_ngrams = None
    else:
        overlapped_ngrams = defaultdict(lambda: defaultdict(int))

    # commpute the stats
    for input_file_index in range(len(input_file_paths)):
        input_file_path: str = input_file_paths[input_file_index]
        print(f"Computing contamination stats for file {input_file_index+1}/{len(input_file_paths)} {input_file_path}")
        file_contamination_stats = compute_scenario_file_contamination(
            scenarios=scenarios,
            training_file_path=input_file_path,
            n_values=N_VALUES,
            file_format=args.input_format,
            ngram_index=ngram_index,
            tokenizer=tokenizer,
            overlapped_ngrams=overlapped_ngrams,
        )
        for stats_repr in all_contamination_stats:
            all_contamination_stats[stats_repr].merge(file_contamination_stats[stats_repr])

    stats_summaries: List[str] = []
    for contamination_stats in all_contamination_stats.values():
        stats_summaries.append(contamination_stats.generate_summary(args.tags))

    with open(args.output_stats, "w") as f:
        f.write("\n".join(stats_summaries))
    print(f"Written {len(stats_summaries)} results to {args.output_stats}")

    if args.output_ngrams is not None:
        with open(args.output_ngrams, "w") as f:
            json.dump(overlapped_ngrams, f)
        print(f"Written the overlapped ngrams to {args.output_ngrams}")
    else:
        print("Overlapped ngrams are not written to disk. Set --output_ngrams if you want output the data.")
