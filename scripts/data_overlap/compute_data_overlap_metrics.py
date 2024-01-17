import json
import os
import glob

from typing import List, Tuple, Set, DefaultDict
from nltk import ngrams
from typing import Dict
from tqdm import tqdm
from collections import defaultdict

from light_scenario import LightInstance, LightScenario, LightScenarioKey
from data_overlap_spec import (
    DataOverlapStats,
    DataOverlapStatsKey,
    OverlapProtocolSpec,
    EntryDataOverlapKey,
    EntryOverlapNgrams,
)
from light_tokenizer import LightTokenizer
from load_documents import get_document_iterator
from common.hierarchical_logger import hlog, htrack_block
from common.general import asdict_without_nones
from common.arguments import get_data_overlap_args
from common.util import get_tokenizer
from scenarios.scenario import ScenarioSpec


PART_INPUT: str = "input"
PART_REF: str = "references"


# type alias for overlap-related data structures
Ngram = Tuple[str, ...]
NgramIndex = Dict[int, Dict[Ngram, Set[EntryDataOverlapKey]]]
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
            input=instance_dict[PART_INPUT], references=instance_dict[PART_REF], id=instance_dict["id"]
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
    stats_key_counts: Dict[DataOverlapStatsKey, int],
) -> NgramIndex:
    """
    Given a list of scenarios and n values, initialize ngram_index.
    stats_key_counts is passed in and updated, counting the number of times a stats_key occurs
    """
    ngram_index: NgramIndex = {n: {} for n in n_values}
    for scenario in light_scenarios:
        hlog(f"Building ngram indexes for {scenario.scenario_key}")
        for n in n_values:
            stats_key = DataOverlapStatsKey(
                light_scenario_key=scenario.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(n=n)
            )
            stats_key_counts[stats_key] = len(scenario.instances)
            for i, instance in enumerate(scenario.instances):
                id = instance.id
                assert id
                input_tokens = tokenizer.tokenize(instance.input)
                for input_ngram in ngrams(input_tokens, n):
                    if input_ngram not in ngram_index[n]:
                        ngram_index[n][input_ngram] = set()
                    ngram_index[n][input_ngram].add(
                        EntryDataOverlapKey(stats_key=stats_key, instance_id=id, part=PART_INPUT)
                    )

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        if reference_ngram not in ngram_index[n]:
                            ngram_index[n][reference_ngram] = set()
                        ngram_index[n][reference_ngram].add(
                            EntryDataOverlapKey(stats_key=stats_key, instance_id=id, part=PART_REF)
                        )
    return ngram_index


def compute_all_data_overlap(
    training_file_path: str,
    file_format: str,
    ngram_index: NgramIndex,
    tokenizer: LightTokenizer,
    stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set[str]],
    stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set[str]],
    entry_overlap_key_to_ngram_counts: DefaultDict[EntryDataOverlapKey, DefaultDict[str, int]],
    output_ngrams: bool,
) -> None:
    """
    Given an input file, compute a overlap stats for each n and each scenario by calling
    `compute_document_data_overlap()` for each document in the file. The function writes
    to the stats_key_to_input_ids and stats_key_to_reference_ids directly and does not return anything.
    training_file_path: file path of the training data
    file_format: format of the training file(s)
    ngram_index: The ngram index that maps from ngrams to overlap stats
    tokenizer: The tokenizer used to break documents in the file into tokens
    stats_key_to_input_ids: a dict mapping the stats key to the overlapping input ids
    stats_key_to_reference_ids: a dict mapping the stats key to the overlapping reference ids
    entry_overlap_key_to_ngram_counts: a dict mapping the key to the overlapping ngrams
    output_ngrams: whether we should output ngrams
    """
    document_iterator = get_document_iterator(file_path=training_file_path, file_format=file_format)
    for document in document_iterator:
        compute_document_data_overlap(
            document=document,
            ngram_index=ngram_index,
            tokenizer=tokenizer,
            stats_key_to_input_ids=stats_key_to_input_ids,
            stats_key_to_reference_ids=stats_key_to_reference_ids,
            entry_overlap_key_to_ngram_counts=entry_overlap_key_to_ngram_counts,
            output_ngrams=output_ngrams,
        )


def compute_document_data_overlap(
    document: str,
    ngram_index: NgramIndex,
    tokenizer: LightTokenizer,
    stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set[str]],
    stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set[str]],
    entry_overlap_key_to_ngram_counts: DefaultDict[EntryDataOverlapKey, DefaultDict[str, int]],
    output_ngrams: bool,
) -> None:
    """
    Given a document, compute a overlap stats for each n and each scenario. The function
    writes to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    tokenizer: The tokenizer used to break the document into tokens

    stats_key_to_input_ids: Dict to keep track of input_ids that are overlapping

    stats_key_to_reference_ids: Dict to keep track of reference_ids that are overlapping

    entry_overlap_key_to_ngram_counts: a dict mapping the key to the overlapping ngrams

    output_ngrams: whether we should output ngrams
    """

    document_tokens = tokenizer.tokenize(document)
    for n in ngram_index.keys():
        for document_ngram in ngrams(document_tokens, n):
            if document_ngram in ngram_index[n]:
                for entry_overlap_key in ngram_index[n][document_ngram]:
                    id = entry_overlap_key.instance_id
                    part = entry_overlap_key.part
                    if part == PART_INPUT:
                        stats_key_to_input_ids[entry_overlap_key.stats_key].add(id)
                    elif part == PART_REF:
                        stats_key_to_reference_ids[entry_overlap_key.stats_key].add(id)
                    if output_ngrams:
                        entry_overlap_key_to_ngram_counts[entry_overlap_key][document_ngram] += 1


if __name__ == "__main__":
    args = get_data_overlap_args()

    tokenizer: LightTokenizer = get_tokenizer(args.normalization)

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

    stats_key_counts: DefaultDict[DataOverlapStatsKey, int] = defaultdict(int)
    with htrack_block("Initializing the stats, ngram_index, and ngram_counter"):
        ngram_index: NgramIndex
        ngram_index = create_ngram_index(
            light_scenarios=light_scenarios, n_values=args.N, tokenizer=tokenizer, stats_key_counts=stats_key_counts
        )

    # DataOverlapStatsKey -> Set[str] for ids
    stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
    stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)

    entry_overlap_key_to_ngram_counts: DefaultDict[EntryDataOverlapKey, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

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
                entry_overlap_key_to_ngram_counts=entry_overlap_key_to_ngram_counts,
                output_ngrams=not args.no_output_ngrams,
            )

    if not args.no_output_ngrams:
        all_entry_overlap_ngrams = []
        with open(f"{args.output_stats}_ngrams", "w") as f:
            for entry_overlap_key in entry_overlap_key_to_ngram_counts:
                ngram_counts = [
                    ngram_count for ngram_count in entry_overlap_key_to_ngram_counts[entry_overlap_key].items()
                ]
                entry_overlap_ngrams = EntryOverlapNgrams(
                    entry_data_overlap_key=entry_overlap_key, overlapping_ngram_counts=ngram_counts
                )
                all_entry_overlap_ngrams.append(entry_overlap_ngrams)
                f.write(f"{json.dumps(asdict_without_nones(entry_overlap_ngrams))}\n")

    all_data_overlap_stats = []
    for stats_key, count in stats_key_counts.items():
        data_overlap_stats = DataOverlapStats(
            data_overlap_stats_key=stats_key,
            instance_ids_with_overlapping_input=sorted(stats_key_to_input_ids[stats_key]),
            instance_ids_with_overlapping_reference=sorted(stats_key_to_reference_ids[stats_key]),
            num_instances=count,
        )
        all_data_overlap_stats.append(data_overlap_stats)

    with open(args.output_stats, "w") as f:
        f.writelines(
            f"{json.dumps(asdict_without_nones(data_overlap_stats))}\n" for data_overlap_stats in all_data_overlap_stats
        )
    hlog(f"Written {len(all_data_overlap_stats)} results to {args.output_stats}")
