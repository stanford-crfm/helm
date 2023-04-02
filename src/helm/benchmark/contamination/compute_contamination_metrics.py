import json
import argparse
import os
import glob

from typing import List, Tuple, Set, Any
from nltk import ngrams
from typing import Dict, Optional, DefaultDict
from collections import defaultdict
from tqdm import tqdm

from helm.benchmark.contamination.light_scenario import LightInstance, LightScenario, LightScenarioKey
from helm.benchmark.contamination.light_tokenizer import LightTokenizer, DefaultTokenizer
from helm.benchmark.contamination.load_documents import get_document_generator
from helm.benchmark.contamination.contamination_stats import (
    ContaminationStats,
    ContaminationStatsKey,
    PART_INPUT,
    PART_REF,
)
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.benchmark.scenarios.scenario import ScenarioSpec


# The n values of the ngrams to be computed
N_VALUES: List[int] = [5, 9, 13]  # TODO: Pick the N values


def load_light_scenarios_from_jsonl(path: str) -> List[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file, where each json represents a LightScenario object.

    Input file format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...

    Each line is a json and each json looks like:
    {
        "light_scenario_key": {
            "metadata":{
                "split": "SPLIT",
                "scenario_attribute_1": "ATTRIBUTE1",
                "scenario_attribute_2": "ATTRIBUTE2",
            }
        },
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

    Note that the values of light_scenario_key.metadata need to be hashable.
    """

    def create_light_instance_from_dict(instance_dict: dict) -> LightInstance:
        return LightInstance(input=instance_dict["input"], references=instance_dict["references"])

    light_scenarios: List[LightScenario] = []
    light_scenario_jsons = open(path, "r").readlines()
    for light_scenario_json in light_scenario_jsons:
        light_scenario_dict: dict = json.loads(light_scenario_json)

        light_scenario_metadata: dict = light_scenario_dict["light_scenario_key"]["metadata"]
        # if the light_scenarios are exported from helm, they will have a scenario_spec field
        if "scenario_spec" in light_scenario_metadata:
            light_scenario_metadata["scenario_spec"] = ScenarioSpec(**light_scenario_metadata["scenario_spec"])
        light_scenario_key = LightScenarioKey(metadata=light_scenario_metadata)
        light_instances: List[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in light_scenario_dict["light_instances"]
        ]
        light_scenarios.append(LightScenario(light_scenario_key=light_scenario_key, light_instances=light_instances))
    return light_scenarios


def create_stats_and_ngram_index(
    light_scenarios: List[LightScenario], n_values: List[int], tokenizer: LightTokenizer
) -> Tuple[Dict[ContaminationStatsKey, ContaminationStats], DefaultDict[int, DefaultDict[Tuple[str], Set[tuple]]]]:
    """Given a list of scenarios and n values, initialize the stats and ngram_index data structures"""
    # mapping from stats repr to stats instance
    all_contamination_stats: Dict[ContaminationStatsKey, ContaminationStats] = {}
    ngram_index: DefaultDict[int, DefaultDict[Tuple[str], Set[tuple]]] = defaultdict(lambda: defaultdict(set))
    for scenario in light_scenarios:
        hlog(f"Building ngram indexes for {scenario.light_scenario_key}")
        for n in n_values:
            # Initizlize a stats instance for every pair of <scenario, n>
            stats: ContaminationStats = ContaminationStats.from_scenario(scenario, stats_tags={"N": n})
            # stats_repr: str = str(stats.stats_spec)
            if stats.stats_key in all_contamination_stats:
                raise ValueError("Duplicated settings detected.")
            all_contamination_stats[stats.stats_key] = stats

            # Build the ngram index
            for i in range(len(scenario.light_instances)):
                instance = scenario.light_instances[i]
                # compute input ngrams
                # TODO: The unigram computation can be taken out to further optimize efficiency if necessary.
                input_unigrams = tokenizer.tokenize(instance.input)
                for input_ngram in ngrams(input_unigrams, n):
                    ngram_index[n][input_ngram].add((stats, i, PART_INPUT))

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        ngram_index[n][reference_ngram].add((stats, i, PART_REF))
    return all_contamination_stats, ngram_index


def compute_scenario_file_contamination(
    training_file_path: str,
    n_values: List[int],
    file_format: str,
    ngram_index: DefaultDict[int, DefaultDict[Tuple[str], set]],
    tokenizer: LightTokenizer,
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]] = None,
):
    """
    Given an input file, compute a contamination stats for each n and each scenario

    ngram_index: The ngram index that maps from ngrams to contamination stats

    tokenizer: The tokenizer used to break documents in the file into tokens

    overlapped_ngrams: The ngrams that are overlapped between the training file and the scenario data and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """
    document_generator = get_document_generator(file_path=training_file_path, file_format=file_format)
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


def compute_scenario_document_contamination(
    ngram_index: Dict[int, DefaultDict[Tuple[str], set]],
    document: str,
    n_values: List[int],
    tokenizer: LightTokenizer,
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]] = None,
):
    """
    Given a document, compute a contamination stats for each n and each scenario

    ngram_index: The ngram index that maps from ngrams to contamination stats

    tokenizer: The tokenizer used to break the document into tokens

    overlapped_ngrams: The ngrams that are overlapped between the training file and the scenario data and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """
    document_tokens = tokenizer.tokenize(document)
    for n in n_values:
        assert n in ngram_index
        for document_ngram in ngrams(document_tokens, n):
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
        nargs="*",
        help="Other tags, such as whether the input data is for pretraining or instruction tuning.",
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
    hlog(f"The input data will be loaded from {input_file_paths}")

    hlog(f"Loading scenario data from {args.scenario_data}")
    light_scenarios = load_light_scenarios_from_jsonl(args.scenario_data)

    with htrack_block("Initializing the stats and ngram_index"):
        all_contamination_stats: Dict[ContaminationStatsKey, ContaminationStats]
        ngram_index: DefaultDict[int, DefaultDict[Tuple[str], Set[tuple]]]
        all_contamination_stats, ngram_index = create_stats_and_ngram_index(light_scenarios, N_VALUES, tokenizer)

    # Initialize overlapped_ngrams
    overlapped_ngrams: Optional[DefaultDict[int, DefaultDict[str, int]]]
    if args.output_ngrams is None:
        overlapped_ngrams = None
    else:
        overlapped_ngrams = defaultdict(lambda: defaultdict(int))

    # commpute the stats
    for input_file_index in tqdm(
        range(len(input_file_paths)), desc="Computing contamination stats for input files", disable=None
    ):
        input_file_path: str = input_file_paths[input_file_index]
        compute_scenario_file_contamination(
            training_file_path=input_file_path,
            n_values=N_VALUES,
            file_format=args.input_format,
            ngram_index=ngram_index,
            tokenizer=tokenizer,
            overlapped_ngrams=overlapped_ngrams,
        )

    stats_summaries: List[Dict[str, Any]] = []
    for contamination_stats in all_contamination_stats.values():
        stats_summaries.append(contamination_stats.generate_summary({"tags:": args.tags}))

    with open(args.output_stats, "w") as f:
        f.writelines(f"{json.dumps(stats_summary)}\n" for stats_summary in stats_summaries)
    hlog(f"Written {len(stats_summaries)} results to {args.output_stats}")

    if args.output_ngrams is not None:
        with open(args.output_ngrams, "w") as f:
            json.dump(overlapped_ngrams, f)
        hlog(f"Written the overlapped ngrams to {args.output_ngrams}")
    else:
        hlog("Overlapped ngrams are not written to disk. Set --output_ngrams if you want output the data.")
