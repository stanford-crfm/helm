import json
import argparse
import os
import glob

from typing import List, Tuple, Set, Any, Optional, DefaultDict
from nltk import ngrams
from typing import Dict
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

from light_scenario import LightInstance, LightScenario, LightScenarioKey
from data_overlap_spec import DataOverlapStats, DataOverlapStatsKey, OverlapProtocolSpec
from light_tokenizer import LightTokenizer, DefaultTokenizer
from load_documents import get_document_iterator
from data_overlap_stats import (
    OldDataOverlapStats,
    OldDataOverlapStatsKey,
    PART_INPUT,
    PART_REF,
)
from common.hierarchical_logger import hlog, htrack_block
from common.general import asdict_without_nones, write
from scenarios.scenario import ScenarioSpec


# The n values of the ngrams to be computed
N_VALUES: List[int] = [5, 9, 13]  # TODO: Pick the N values


@dataclass(frozen=True)
class EntryDataOverlapKey:
    """Unique key representing either the input or references of a single instance in a scenario."""

    stats_key: DataOverlapStatsKey
    part: str
    """Either PART_INPUT or PART_REF"""
    instance_id: str
    index: int


# type alias for overlap-related data structures
Ngram = Tuple[str, ...]
NgramIndex = Dict[int, Dict[Ngram, Set[EntryDataOverlapKey]]]
AllDataOverlapStats = Dict[OldDataOverlapStatsKey, OldDataOverlapStats]
NgramCounter = Dict[EntryDataOverlapKey, Dict[Ngram, int]]


def load_light_scenarios_from_jsonl(path: str) -> List[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file, where each json represents a LightScenario object.

    Input file format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...
    """

    def create_light_instance_from_dict(instance_dict: dict) -> LightInstance:
        return LightInstance(
            input=instance_dict["input"], references=instance_dict["references"], id=instance_dict["id"]
        )

    light_scenarios: List[LightScenario] = []
    light_scenario_jsons = open(path, "r").readlines()
    for light_scenario_json in light_scenario_jsons:
        light_scenario_dict: dict = json.loads(light_scenario_json)

        light_scenario_key_dict: dict = light_scenario_dict["scenario_key"]
        # if the light_scenarios are exported from helm, they will have a scenario_spec field
        scenario_spec = ScenarioSpec(**light_scenario_key_dict["scenario_spec"])
        light_scenario_key = LightScenarioKey(scenario_spec=scenario_spec, split=light_scenario_key_dict["split"])
        light_instances: List[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in light_scenario_dict["instances"]
        ]
        light_scenarios.append(LightScenario(scenario_key=light_scenario_key, instances=light_instances))
    return light_scenarios


def create_ngram_index(
    light_scenarios: List[LightScenario],
    n_values: List[int],
    tokenizer: LightTokenizer,
    stats_keys: Set[DataOverlapStatsKey],
) -> NgramIndex:
    """Given a list of scenarios and n values, initialize ngram_index"""
    ngram_index: NgramIndex = {n: {} for n in n_values}
    for scenario in light_scenarios:
        hlog(f"Building ngram indexes for {scenario.scenario_key}")
        for n in n_values:
            stats_key = DataOverlapStatsKey(
                light_scenario_key=scenario.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(N=n)
            )
            stats_keys.add(stats_key)
            for i in range(len(scenario.instances)):
                instance = scenario.instances[i]
                input_tokens = tokenizer.tokenize(instance.input)
                for input_ngram in ngrams(input_tokens, n):
                    if input_ngram not in ngram_index[n]:
                        ngram_index[n][input_ngram] = set()
                    id = instance.id if instance.id else f"id{i}"
                    ngram_index[n][input_ngram].add(
                        EntryDataOverlapKey(stats_key=stats_key, instance_id=id, index=i, part=PART_INPUT)
                    )

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        if reference_ngram not in ngram_index[n]:
                            ngram_index[n][reference_ngram] = set()
                        id = instance.id if instance.id else f"id{i}"
                        ngram_index[n][reference_ngram].add(
                            EntryDataOverlapKey(stats_key=stats_key, instance_id=id, index=i, part=PART_REF)
                        )
    return ngram_index


def create_all_data_overlap_stats(light_scenarios: List[LightScenario], n_values: List[int]) -> AllDataOverlapStats:
    """Given a list of scenarios and n values, initialize all_overlap_stats"""
    hlog("Initializing all data overlap stats")
    all_overlap_stats: AllDataOverlapStats = {}
    for scenario in light_scenarios:
        for n in n_values:
            # Initialize a stats instance for every pair of <scenario, n>
            stats: OldDataOverlapStats = OldDataOverlapStats.from_scenario(scenario, stats_tags={"N": n})
            if stats.stats_key in all_overlap_stats:
                raise ValueError("Duplicated settings detected.")
            all_overlap_stats[stats.stats_key] = stats
    return all_overlap_stats


def compute_all_data_overlap(
    training_file_path: str,
    file_format: str,
    ngram_index: NgramIndex,
    tokenizer: LightTokenizer,
    stats_key_to_input_ids: Dict[DataOverlapStatsKey, Set[str]],
    stats_key_to_reference_ids: Dict[DataOverlapStatsKey, Set[str]],
):
    """
    Given an input file, compute a overlap stats for each n and each scenario by calling
    `compute_scenario_document_overlap()` for each document in the file. The function writes
    to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    all_overlap_stats: The overlap stats for each scenario and n. The variable to write to.

    tokenizer: The tokenizer used to break documents in the file into tokens

    ngram_counter: The ngrams that are overlapped between the training file and the scenario data
    and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """
    document_iterator = get_document_iterator(file_path=training_file_path, file_format=file_format)
    for document in document_iterator:
        compute_document_data_overlap(
            document=document,
            ngram_index=ngram_index,
            tokenizer=tokenizer,
            stats_key_to_input_ids=stats_key_to_input_ids,
            stats_key_to_reference_ids=stats_key_to_reference_ids,
        )


def compute_document_data_overlap(
    document: str,
    ngram_index: NgramIndex,
    tokenizer: LightTokenizer,
    stats_key_to_input_ids: Dict[DataOverlapStatsKey, Set[str]],
    stats_key_to_reference_ids: Dict[DataOverlapStatsKey, Set[str]],
):
    """
    Given a document, compute a overlap stats for each n and each scenario. The function
    writes to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    tokenizer: The tokenizer used to break the document into tokens

    all_overlap_stats: The overlap stats for each scenario and n. The variable to write to.

    ngram_counter: The ngrams that are overlapped between the training file and the scenario data
    and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """

    document_tokens = tokenizer.tokenize(document)
    for n in ngram_index.keys():
        for document_ngram in ngrams(document_tokens, n):
            if document_ngram in ngram_index[n]:
                for entry_overlap_key in ngram_index[n][document_ngram]:
                    id = entry_overlap_key.instance_id
                    part = entry_overlap_key.part
                    if part == "input":
                        stats_key_to_input_ids[entry_overlap_key.stats_key].add(id)
                    elif part == "reference":
                        stats_key_to_reference_ids[entry_overlap_key.stats_key].add(id)
                    else:
                        hlog("Part neither input nor reference, hence not recording")


def compute_scenario_file_data_overlap(
    training_file_path: str,
    file_format: str,
    ngram_index: NgramIndex,
    all_overlap_stats: AllDataOverlapStats,
    tokenizer: LightTokenizer,
    ngram_counter: Optional[NgramCounter] = None,
    max_overlapping_ngrams: int = 0,
):
    """
    Given an input file, compute a overlap stats for each n and each scenario by calling
    `compute_scenario_document_overlap()` for each document in the file. The function writes
    to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    all_overlap_stats: The overlap stats for each scenario and n. The variable to write to.

    tokenizer: The tokenizer used to break documents in the file into tokens

    ngram_counter: The ngrams that are overlapped between the training file and the scenario data
    and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """
    document_iterator = get_document_iterator(file_path=training_file_path, file_format=file_format)
    for document in document_iterator:
        compute_scenario_document_data_overlap(
            document=document,
            ngram_index=ngram_index,
            all_overlap_stats=all_overlap_stats,
            tokenizer=tokenizer,
            ngram_counter=ngram_counter,
            max_overlapping_ngrams=max_overlapping_ngrams,
        )


def compute_scenario_document_data_overlap(
    document: str,
    ngram_index: NgramIndex,
    all_overlap_stats: AllDataOverlapStats,
    tokenizer: LightTokenizer,
    ngram_counter: Optional[NgramCounter] = None,
    max_overlapping_ngrams: int = 0,
):
    """
    Given a document, compute a overlap stats for each n and each scenario. The function
    writes to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    tokenizer: The tokenizer used to break the document into tokens

    all_overlap_stats: The overlap stats for each scenario and n. The variable to write to.

    ngram_counter: The ngrams that are overlapped between the training file and the scenario data
    and their counts.
    The outer dict maps from n to the inner dict, which maps from ngram to count.
    """
    document_tokens = tokenizer.tokenize(document)
    for n in ngram_index.keys():
        for document_ngram in ngrams(document_tokens, n):
            if document_ngram in ngram_index[n]:
                for entry_overlap_key in ngram_index[n][document_ngram]:
                    # update overlap_stats
                    stats: OldDataOverlapStats = all_overlap_stats[entry_overlap_key.stats_key]
                    stats.write_one_to_bit(entry_overlap_key.index, entry_overlap_key.part)
                    # skip the rest if max_overlapping_ngrams is 0
                    if max_overlapping_ngrams != 0:
                        if ngram_counter is None:
                            raise ValueError("ngram_counter must be not none when max_overlapping_ngrams != 0")
                        # update ngram_counter
                        if entry_overlap_key not in ngram_counter:
                            ngram_counter[entry_overlap_key] = {}
                        if document_ngram in ngram_counter[entry_overlap_key]:
                            ngram_counter[entry_overlap_key][document_ngram] += 1
                        elif (
                            max_overlapping_ngrams == -1
                            or len(ngram_counter[entry_overlap_key]) < max_overlapping_ngrams
                        ):
                            ngram_counter[entry_overlap_key][document_ngram] = 1


def get_all_data_overlap_stats(
    ngram_index: NgramIndex,
):
    """
    ngram_index: The ngram index that maps from ngrams to overlap stats
    """
    stats_keys = set()
    stats_key_to_input_ids = defaultdict(set)
    stats_key_to_reference_ids = defaultdict(set)
    for n in ngram_index.keys():  # these are the n, say [5, 9, 13]
        for _, overlap_keys in ngram_index[n].items():
            for overlap_key in overlap_keys:
                light_scenario_key = overlap_key.stats_key.metadata["light_scenario_key"]
                overlap_protocol_spec = OverlapProtocolSpec(N=n)
                data_overlap_stats_key = DataOverlapStatsKey(
                    light_scenario_key=light_scenario_key,
                    overlap_protocol_spec=overlap_protocol_spec,
                )
                stats_keys.add(data_overlap_stats_key)
                id = overlap_key.instance_id
                part = overlap_key.part
                if part == "input":
                    stats_key_to_input_ids[data_overlap_stats_key].add(id)
                elif part == "reference":
                    stats_key_to_reference_ids[data_overlap_stats_key].add(id)
                else:
                    hlog("Part neither input nor reference, hence not recording")
    all_data_overlap_stats = []
    for stats_key in stats_keys:
        data_overlap_stats = OldDataOverlapStats(
            data_overlap_stats_key=stats_key,
            instance_ids_with_overlapping_input=sorted(stats_key_to_input_ids[data_overlap_stats_key]),
            instance_ids_with_overlapping_reference=sorted(stats_key_to_reference_ids[data_overlap_stats_key]),
        )
        all_data_overlap_stats.append(data_overlap_stats)
    return all_data_overlap_stats


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
        help="Other tags, such as whether the input data is for pretraining or instruction tuning",
    )
    parser.add_argument(
        "--normalization", type=str, default="default", help="What normalization and tokenization strategy to apply"
    )
    parser.add_argument(
        "--output-ngrams",
        type=str,
        default=None,
        help="Path to the file of overlapping ngrams. To output the ngrams, you must also specify --max-output-ngrams",
    )
    parser.add_argument(
        "--max-output-ngrams",
        type=int,
        default=0,
        help=(
            "The max number of overlapping ngrams to be stored for each (n, light_instance, part)."
            "Set to -1 to store all"
        ),
    )

    args = parser.parse_args()

    tokenizer: LightTokenizer
    if args.normalization == "none":
        tokenizer = LightTokenizer()
    elif args.normalization == "default":
        tokenizer = DefaultTokenizer()
    else:
        raise ValueError(f"Normalization strategy {args.normalization} is not defined.")

    if args.max_output_ngrams != 0 and args.output_ngrams is None:
        raise ValueError("You must specify --output-ngrams if you want to output ngrams.")
    if args.max_output_ngrams == 0 and args.output_ngrams is not None:
        raise ValueError("You must specify --max-output-ngrams != 0 if you want to output ngrams.")

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

    stats_keys: Set[DataOverlapStatsKey] = set()
    with htrack_block("Initializing the stats, ngram_index, and ngram_counter"):
        ngram_index: NgramIndex
        ngram_index = create_ngram_index(
            light_scenarios=light_scenarios, n_values=N_VALUES, tokenizer=tokenizer, stats_keys=stats_keys
        )

    # DataOverlapStatsKey -> Set[str] for ids
    stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
    stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)

    # commpute the stats
    with htrack_block("Computing overlap stats"):
        for input_file_index in tqdm(
            range(len(input_file_paths)), desc="Computing overlap stats for input files", disable=None
        ):
            input_file_path: str = input_file_paths[input_file_index]
            compute_all_data_overlap(
                training_file_path=input_file_path,
                file_format=args.input_format,
                ngram_index=ngram_index,
                tokenizer=tokenizer,
                stats_key_to_input_ids=stats_key_to_input_ids,
                stats_key_to_reference_ids=stats_key_to_reference_ids,
            )

    all_data_overlap_stats = []
    for stats_key in stats_keys:
        data_overlap_stats = DataOverlapStats(
            data_overlap_stats_key=stats_key,
            instance_ids_with_overlapping_input=sorted(stats_key_to_input_ids[stats_key]),
            instance_ids_with_overlapping_reference=sorted(stats_key_to_reference_ids[stats_key]),
        )
        all_data_overlap_stats.append(data_overlap_stats)

    with open(args.output_stats, "w") as f:
        f.writelines(
            f"{json.dumps(asdict_without_nones(data_overlap_stats))}\n" for data_overlap_stats in all_data_overlap_stats
        )
    hlog(f"Written {len(all_data_overlap_stats )} results to {args.output_stats}")

    if args.output_ngrams is not None:
        all_overlap_stats = create_all_data_overlap_stats(light_scenarios=light_scenarios, n_values=N_VALUES)
        ngram_counter: NgramCounter = {}

        # commpute the stats
        with htrack_block("Computing overlap stats"):
            for input_file_index in tqdm(
                range(len(input_file_paths)), desc="Computing overlap stats for input files", disable=None
            ):
                compute_scenario_file_data_overlap(
                    training_file_path=input_file_path,
                    file_format=args.input_format,
                    ngram_index=ngram_index,
                    all_overlap_stats=all_overlap_stats,
                    tokenizer=tokenizer,
                    ngram_counter=ngram_counter,
                    max_overlapping_ngrams=args.max_output_ngrams,
                )

        stats_summaries: List[Dict[str, Any]] = []
        for overlap_stats in all_overlap_stats.values():
            stats_summaries.append(overlap_stats.generate_summary({"tags:": args.tags}))

        with open(args.output_stats, "w") as f:
            f.writelines(f"{json.dumps(stats_summary)}\n" for stats_summary in stats_summaries)
        hlog(f"Written {len(stats_summaries)} results to {args.output_stats}")

        # convert the ngram counter to json format
        ngram_entries = []
        for entry_overlap_key, overlapping_ngrams in ngram_counter.items():
            ngram_entries.append(
                {
                    "entry_overlap_key": asdict_without_nones(entry_overlap_key),
                    "overlapping_ngrams": {" ".join(ngram): count for ngram, count in overlapping_ngrams.items()},
                }
            )
        write(args.output_ngrams, json.dumps(ngram_entries))
    else:
        hlog(
            "Overlapping ngrams are not written to disk. "
            "Set --output-ngrams and --max-output-ngrams if you want output the data."
        )
