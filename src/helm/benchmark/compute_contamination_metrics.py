import json
import argparse
import os
import glob
from dataclasses import dataclass
from typing import List
from bitarray import bitarray
from nltk import ngrams
from typing import Dict, Optional, Generator, DefaultDict
from collections import defaultdict

# The n values of the ngrams to be computed
N_VALUES: List[int] = [5, 9, 13]  # TODO: Pick the N values

PART_INPUT: str = "input"
PART_REF: str = "reference"


@dataclass(frozen=True, eq=False)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: List[str]
    """References that help us evaluate"""


@dataclass(frozen=True, eq=False)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_spec: str
    """The scenario spec"""

    light_instances: List[LightInstance]
    """Instances of this scenario"""


@dataclass(frozen=True, eq=False)
class DocumentReadingProcessor:
    """
    TODO: This is probably a bad class name.
    """

    file_path: str
    """The path to the data file"""

    file_format: str
    """The type of the file format, which determines how it will be read"""

    def get_document_generator(self) -> Generator:
        if self.file_format == "the_pile":
            return self.get_the_pile_document_generator()
        elif self.file_format == "raw":
            return self.get_raw_document_generator()
        elif self.file_format == "custom":
            return self.get_custom_document_generator()
        else:
            raise NotImplementedError()

    def get_the_pile_document_generator(self) -> Generator:
        """
        This method reads input files with similar file formats with The Pile's jsonl format.
        Each line of the input file should be a json string, where the document is stored in a field named "text".
        There are no empty lines between json lines.

        Example:
        {"text": "Hello World!", "meta": {"pile_set_name": "Pile-CC"}}
        {"text": "Foo bar", "meta": {"pile_set_name": "Pile-CC"}}
        """
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)["text"]

    def get_raw_document_generator(self) -> Generator:
        """
        This method reads input files where each line is a document. The file should not be organized
        in any specific file structures such as json, jsonl, or tsv, as this may affect ngram computation.
        Any characters other than the actual text content should be removed.

        Example:
        Hello World!
        Foo bar
        This is the 3rd document.
        """
        with open(self.file_path, "r") as f:
            for line in f:
                yield line.rstrip("\n")

    def get_custom_document_generator(self) -> Generator:
        """Define your own document reading method"""
        pass


class ContaminationStats:
    """
    A memory-efficient class for contamination stats. The core data structures are bit arrays where
    every bit records whether an instance is dirty (contaminated) or not.
    """

    def __init__(self, scenario_spec: str, num_instances: int, stats_tags: Optional[List[str]] = None):
        self.scenario_spec = scenario_spec
        self.num_instances = num_instances
        self.stats_tags: List[str]
        if isinstance(stats_tags, list):
            self.stats_tags = stats_tags
        else:
            self.stats_tags = []

        self._input_bits = bitarray(num_instances)
        self._reference_bits = bitarray(num_instances)
        self._input_bits.setall(0)
        self._reference_bits.setall(0)

    @classmethod
    def from_scenario(cls, scenario: LightScenario, stats_tags: Optional[List[str]] = None):
        return cls(
            scenario_spec=scenario.scenario_spec, num_instances=len(scenario.light_instances), stats_tags=stats_tags
        )

    def write_dirty(self, instance_id: int, part: str):
        if part == PART_INPUT:
            self._input_bits[instance_id] = 1
        elif part == PART_REF:
            self._reference_bits[instance_id] = 1
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def get_bit(self, instance_id: int, part: str) -> int:
        if part == PART_INPUT:
            return self._input_bits[instance_id]
        elif part == PART_REF:
            return self._reference_bits[instance_id]
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def merge(self, stats):
        """Merge two stats instance of the same scenario"""
        if self.scenario_spec != stats.scenario_spec:
            raise ValueError("Only stats for the same scenario can be merged.")
        if self.num_instances != stats.num_instances:
            raise ValueError("The sizes of the two scenarios need to equal.")
        self._input_bits |= stats._input_bits
        self._reference_bits |= stats._reference_bits

    @property
    def num_input_positive_instances(self):
        return self._input_bits.count()

    @property
    def num_reference_positive_instances(self):
        return self._reference_bits.count()

    @property
    def input_positive_rate(self):
        return self._input_bits.count() / self.num_instances

    @property
    def reference_positive_rate(self):
        return self._reference_bits.count() / self.num_instances

    @property
    def stats_repr(self) -> str:
        return f"{self.scenario_spec},{','.join(self.stats_tags)}"

    def generate_summary(self, tags: List[str]) -> str:
        """Output a summary of the stats"""
        summary = {
            "setting": f"{self.stats_repr}{'' if len(tags) == 0 else ','+','.join(tags)}",
            "total_instances": self.num_instances,
            "num_input_positive_instances": self.num_input_positive_instances,
            "num_reference_positive_instances": self.num_reference_positive_instances,
            "input_positive_rate": self.input_positive_rate,
            "reference_positive_rate": self.reference_positive_rate,
        }
        return json.dumps(summary)


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
        )

    return all_scenario_stats


def compute_scenario_document_contamination(
    ngram_index: Dict[int, DefaultDict[str, list]],
    document: str,
    n_values: List[int],
):
    """Given a document, compute a contamination stats for each n and each scenario"""
    document_unigrams = document.split()
    for n in n_values:
        assert n in ngram_index
        for document_ngram in ngrams(document_unigrams, n):
            if document_ngram in ngram_index[n]:
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

    args = parser.parse_args()

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
                input_unigrams = instance.input.split()
                for input_ngram in ngrams(input_unigrams, n):
                    ngram_index[n][input_ngram].append((stats, i, PART_INPUT))

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = reference.split()
                    for reference_ngram in ngrams(reference_unigrams, n):
                        ngram_index[n][reference_ngram].append((stats, i, PART_REF))

    # commpute the stats
    for input_file_path in input_file_paths:
        print(f"Computing contamination stats for {input_file_path}")
        file_contamination_stats = compute_scenario_file_contamination(
            scenarios=scenarios,
            training_file_path=input_file_path,
            n_values=N_VALUES,
            file_format=args.input_format,
            ngram_index=ngram_index,
        )
        for stats_repr in all_contamination_stats:
            all_contamination_stats[stats_repr].merge(file_contamination_stats[stats_repr])

    stats_summaries: List[str] = []
    for contamination_stats in all_contamination_stats.values():
        stats_summaries.append(contamination_stats.generate_summary(args.tags))

    with open(args.output_stats, "w") as f:
        f.write("\n".join(stats_summaries))
    print(f"Written {len(stats_summaries)} results to {args.output_stats}")
