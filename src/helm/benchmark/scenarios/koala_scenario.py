import json
import os
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT, ScenarioMetadata


class KoalaScenario(Scenario):
    """
    This scenario is based on the prompts used by the Koala team to evaluate instruction-following models.

    https://bair.berkeley.edu/blog/2023/04/03/koala/
    """

    name = "koala"
    description = "Koala eval dataset"
    tags = ["instructions"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        source_url = "https://raw.githubusercontent.com/arnav-gudibande/koala-test-set/main/koala_test_set.jsonl"
        data_path: str = os.path.join(output_path, "Koala_prompts.jsonl")

        ensure_file_downloaded(
            source_url=source_url,
            target_path=data_path,
        )

        # Read all the instances
        instances: List[Instance] = []
        for line in open(data_path):
            # Example: {"id": "koala_1", "prompt": "Take MLK speech \"I had a dream\" but turn it into a top 100 rap song"} # noqa: E501
            raw = json.loads(line)
            instance = Instance(
                input=Input(text=raw["prompt"]),
                references=[],
                split=TEST_SPLIT,
            )
            instances.append(instance)
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="koala",
            display_name="Koala test dataset",
            short_display_name="Koala test dataset",
            description="The test dataset from the [Koala "
            "paper](https://bair.berkeley.edu/blog/2023/04/03/koala/) for evaluating "
            "instruction-following models.",
            taxonomy=TaxonomyInfo(
                task="open-ended instruction following",
                what="Instructions for LLMs",
                when="Before 2023",
                who="Web users",
                language="English",
            ),
            main_metric="Helpfulness",
            main_split="test",
        )
