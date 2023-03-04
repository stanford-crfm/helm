import json
import argparse
from dataclasses import dataclass
from typing import List
from bitarray import bitarray
from nltk import ngrams
from typing import Dict, Set, Optional, Generator, DefaultDict
from collections import defaultdict


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

    name: str
    """The scenario name"""

    light_instances: List[LightInstance]
    """Instances of this scenario"""


@dataclass(frozen=True, eq=False)
class ScenarioNgrams:
    """
    A data class that stores the ngram features of a scenario.
    """

    # TODO: explain the structure
    input_ngrams: List[Dict[int, Set[str]]]
    reference_ngrams: List[Dict[int, Set[str]]]


@dataclass(frozen=True, eq=False)
class DocumentReadingProcessor:
    """
    TODO: This is probably a bad class name.
    TODO: make file_format an commandline argument.
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
        # TODO: add an example
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)["text"]

    def get_raw_document_generator(self) -> Generator:
        # TODO: add an example
        with open(self.file_path, "r") as f:
            for line in f:
                yield line.rstrip("\n")

    def get_custom_document_generator(self) -> Generator:
        """Define your own document reading method"""
        pass


class BinaryScenarioMetric:
    """
    A memory-efficient class for contamination metrics. The core data structures are bit arrays where
    every bit records whether an instance is dirty (contaminated) or not.
    """

    def __init__(self, scenario_name: str, num_instances: int, metric_tags: Optional[List[str]] = None):
        self.scenario_name = scenario_name
        self.num_instances = num_instances
        self.metric_tags: List[str]
        if isinstance(metric_tags, list):
            self.metric_tags = metric_tags
        else:
            self.metric_tags = []

        self._input_bits = bitarray(num_instances)
        self._reference_bits = bitarray(num_instances)
        self._input_bits.setall(0)
        self._reference_bits.setall(0)

    @classmethod
    def from_scenario(cls, scenario: LightScenario, metric_tags: Optional[List[str]] = None):
        return cls(scenario_name=scenario.name, num_instances=len(scenario.light_instances), metric_tags=metric_tags)

    def write_dirty(self, instance_id: int, part: str):
        if part == "input":
            self._input_bits[instance_id] = 1
        elif part == "ref":
            self._reference_bits[instance_id] = 1
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def get_bit(self, instance_id: int, part: str) -> int:
        if part == "input":
            return self._input_bits[instance_id]
        elif part == "ref":
            return self._reference_bits[instance_id]
        else:
            raise ValueError(f"There is no valid part of instance named {part}")

    def merge(self, metric):
        """Merge two metric instance of the same scenario"""
        if self.scenario_name != metric.scenario_name:
            raise ValueError("Only metrics for the same scenario can be merged.")
        if self.num_instances != metric.num_instances:
            raise ValueError("The sizes of the two scenarios need to equal.")
        self._input_bits |= metric._input_bits
        self._reference_bits |= metric._reference_bits

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

    def generate_summary(self, tags) -> str:
        """Output a summary of the metric"""
        summary = {
            "Setting": f"{self.scenario_name},{','.join(tags+self.metric_tags)}",
            "total_instances": self.num_instances,
            "num_input_positive_instances": self.num_input_positive_instances,
            "num_reference_positive_instances": self.num_reference_positive_instances,
            "input_positive_rate": self.input_positive_rate,
            "reference_positive_rate": self.reference_positive_rate,
        }
        return json.dumps(summary)


def load_scenarios_from_jsonl(filename: str) -> List[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file.

    Input File Format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...

    Each line is a json and each json looks like:
    {
        "name": "SCENARIO_NAME",
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
    scenario_jsons = open(filename, "r").readlines()
    for scenario_json in scenario_jsons:
        scenario_dict: dict = json.loads(scenario_json)
        scenario_name: str = scenario_dict["name"]
        light_instances: List[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in scenario_dict["light_instances"]
        ]
        light_scenarios.append(LightScenario(name=scenario_name, light_instances=light_instances))
    return light_scenarios


def compute_scenario_ngrams(scenario: LightScenario, n_values: List[int]):
    """For each n value and each instance, compute the ngram features"""
    # TODO: add an example in the docstring
    # TODO: variable naming
    input_ngrams: List[Dict[int, Set[str]]] = []
    reference_ngrams: List[Dict[int, Set[str]]] = []
    for instance in scenario.light_instances:
        # compute input ngrams
        input_unigrams = instance.input.split()
        input_ngram_dict: Dict[int, Set[str]] = {}
        for n in n_values:
            input_ngram_set = set()
            for ngram in ngrams(input_unigrams, n):
                input_ngram_set.add(ngram)
            input_ngram_dict[n] = input_ngram_set

        # compute reference ngrams
        reference_ngram_dict: Dict[int, Set[str]] = {}
        for n in n_values:
            reference_ngram_set = set()
            for reference in instance.references:
                reference_unigrams = reference.split()
                for ngram in ngrams(reference_unigrams, n):
                    reference_ngram_set.add(ngram)
            reference_ngram_dict[n] = reference_ngram_set

        input_ngrams.append(input_ngram_dict)
        reference_ngrams.append(reference_ngram_dict)
    return ScenarioNgrams(input_ngrams=input_ngrams, reference_ngrams=reference_ngrams)


def compute_instance_contamination(
    scenarios: List[LightScenario], training_file_path: str, n_values: List[int]
) -> List[BinaryScenarioMetric]:
    # TODO: multiple documents

    """For each n, for each scenario, compute a contamination metric"""

    # Initizlize a metric instance for every pair of <scenario, n>
    all_scenario_metrics: DefaultDict[int, dict] = defaultdict(dict)
    for n in n_values:
        for scenario in scenarios:
            all_scenario_metrics[n][scenario.name] = BinaryScenarioMetric.from_scenario(
                scenario, metric_tags=[f"N={n}"]
            )

    # Compute ngrams for each scenario
    all_scenario_ngrams = {
        scenario.name: compute_scenario_ngrams(scenario=scenario, n_values=n_values) for scenario in scenarios
    }
    # TODO: Take it out: not once per file.

    document_generator = DocumentReadingProcessor(
        file_path=training_file_path, file_format="the_pile"
    ).get_document_generator()

    t: int = 0
    # TODO: take the computation part out.
    for document in document_generator:
        t += 1
        print(f"Processing the {t}th document...", end="\r")
        document_unigrams = document.split()
        for scenario in scenarios:
            scenario_ngrams = all_scenario_ngrams[scenario.name]
            for n in n_values:
                scenario_metric = all_scenario_metrics[n][scenario.name]
                document_ngrams = set(ngrams(document_unigrams, n))
                for i in range(len(scenario.light_instances)):
                    if len(document_ngrams.intersection(scenario_ngrams.input_ngrams[i][n])) > 0:
                        scenario_metric.write_dirty(i, "input")
                    if len(document_ngrams.intersection(scenario_ngrams.reference_ngrams[i][n])) > 0:
                        scenario_metric.write_dirty(i, "ref")

    # Flatten the dict
    output_metrics: List[BinaryScenarioMetric] = []
    for n in n_values:
        for scenario in scenarios:
            output_metrics.append(all_scenario_metrics[n][scenario.name])
    return output_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="Path to your training data")
    parser.add_argument("--scenario-data", type=str, required=True, help="Path to scenario data (benchmarking data)")
    parser.add_argument("--output-stats", type=str, required=True, help="Path to the output file")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Other tags, such as whether the input data is for pretraining or instruction tuning",
        default=[],
    )

    args = parser.parse_args()

    print(f"Loading scenario data from {args.scenario_data}")
    scenarios = load_scenarios_from_jsonl(args.scenario_data)

    N_VALUES = [5, 9, 13]  # TODO: Pick the N values

    stats: List[str] = []
    contamination_metrics = compute_instance_contamination(scenarios, args.input_data, n_values=N_VALUES)
    for contamination_metric in contamination_metrics:
        stats.append(contamination_metric.generate_summary(args.tags))

    with open(args.output_stats, "w") as f:
        f.write("\n".join(stats))
    print(f"Written {len(stats)} results to {args.output_stats}")
