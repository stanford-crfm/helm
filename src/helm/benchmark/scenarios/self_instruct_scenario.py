import json
import os
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Reference,
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
    Output,
    ScenarioMetadata,
)


class SelfInstructScenario(Scenario):
    """
    This scenario is based on the manually-curated instructions from the self-instruct paper:

    https://arxiv.org/pdf/2212.10560.pdf

    Note that we are not using the self-instruct method here, just the manual data.
    """

    name = "self-instruct"
    description = "Self-instruct eval dataset"
    tags = ["instructions"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        source_url = "https://raw.githubusercontent.com/yizhongw/self-instruct/main/human_eval/user_oriented_instructions.jsonl"  # noqa: E501
        tasks_path: str = os.path.join(output_path, "user_oriented_instructions.jsonl")

        ensure_file_downloaded(
            source_url=source_url,
            target_path=tasks_path,
        )

        # Read all the instances
        instances: List[Instance] = []
        for line in open(tasks_path):
            # Example: {"id": "seed_task_0", "name": "breakfast_suggestion", "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?", "instances": [{"input": "", "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}], "is_classification": false}  # noqa: E501
            raw = json.loads(line)
            instruction = raw["instruction"]
            for example in raw["instances"]:
                input_text = instruction + "\n" + example["input"]
                instance = Instance(
                    input=Input(text=input_text),
                    references=[
                        Reference(Output(text=example["output"]), tags=[CORRECT_TAG]),
                    ],
                    split=TEST_SPLIT,
                )
                instances.append(instance)
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="self_instruct",
            display_name="Self Instruct",
            short_display_name="Self Instruct",
            description="The manually-curated instructions from the Self-Instruct paper ([Wang et al., "
            "2023](https://aclanthology.org/2023.acl-long.754.pdf)).",
            taxonomy=TaxonomyInfo(
                task="open-ended instruction following",
                what="Instructions for LLMs",
                when="2022",
                who="Authors of the research paper",
                language="English",
            ),
            main_metric="Helpfulness",
            main_split="test",
        )
