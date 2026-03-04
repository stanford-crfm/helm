import os
from typing import List

import datasets

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)


class ArabicContentGenerationScenario(Scenario):
    name = "arabic_content_generation"
    description = "Arabic Content Generation"
    tags = ["content generation"]

    def __init__(self, category: str):
        super().__init__()
        self.category = category

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "yifanmai/arabic-enterprise",
            "content_generation",
            cache_dir=cache_dir,
            split="test",
            # TODO: Set revision
        )
        assert isinstance(dataset, datasets.Dataset)

        instances: List[Instance] = []
        for row in dataset:
            if self.category != "all" and self.category != row["category"]:
                continue
            prompt = f"## Facts\n\n{row['facts']}\n\n## Style\n\n{row['style']}"
            instance = Instance(
                id=f"id{row['id']}",
                input=Input(text=prompt),
                references=[
                    Reference(Output(text=row["text"].strip()), tags=[CORRECT_TAG]),
                ],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="Arabic Content Generation",
            description=self.description,
            main_metric="exact_match",
            main_split="test",
            taxonomy=TaxonomyInfo(
                task="content generation",
                what="news articles",
                who="journalists",
                when="before 2026",
                language="Arabic",
            ),
        )
