from typing import List
from datasets import load_dataset

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output, ScenarioMetadata


class SimpleSafetyTestsScenario(Scenario):
    """
    The test suite comprises 100 test prompts across five harm areas that LLMs,
    for the vast majority of applications, should refuse to comply with
    https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
    """

    name = "SimpleSafetyTests"
    description = "SimpleSafetyTests dataset"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # read explicit data
        dataset = load_dataset(
            "Bertievidgen/SimpleSafetyTests", split="test", revision="98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5"
        )

        # Read all the instances
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["prompt"])
            references = [
                Reference(output=Output(text=row[column_name]), tags=[]) for column_name in ["category", "harm_area"]
            ]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="simple_safety_tests",
            display_name="SimpleSafetyTests",
            description="SimpleSafetyTests",
            taxonomy=TaxonomyInfo(task="question answering", what="n/a", when="n/a", who="n/a", language="English"),
            main_metric="safety_score",
            main_split="test",
        )
