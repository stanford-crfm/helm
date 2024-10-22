import json
import os
from typing import List, Dict
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Reference, Output


DATA_REPO_HASH = "38972f6ccbf376a8d0660babafb4d2b3b9cca3f4"


class DecodingTrustStereotypeBiasScenario(Scenario):
    """
    The DecodingTrustStereotypePrompts dataset is from the paper:
    https://arxiv.org/abs/2306.11698

    The dataset contains 1,152 manually crafted stereotype user prompts.
    The prompts cover 16 stereotype topics (for e.g., drug addiction, HIV, etc.),
    and 24 demographic groups spanning 7 demographic factors: race/ethnicity (Asians, Black
    people, etc.), gender/sexual orientation (homosexuals, men, and women),
    nationality (Mexicans, Americans, etc.), age (old and young people), religion (Muslims,
    Jews, etc.), disability (physically disabled and able-bodied people), and socioeconomic status (poor
    and rich people).
    """

    name = "decodingtrust_stereotype_bias"
    description = (
        "Stereotype bias analysis of LM generations when fed manually crafted user prompts that contains stereotypes."
    )
    tags = ["harms", "stereotype", "bias"]

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "stereotype_bias_data.jsonl")
        ensure_file_downloaded(
            source_url=f"https://raw.githubusercontent.com/AI-secure/DecodingTrust-Data-Legacy/{DATA_REPO_HASH}/data/"
            f"stereotype/dataset/stereotype_bias_data.jsonl",
            target_path=data_path,
            unpack=False,
        )

        instances: List[Instance] = []
        prompts_path: str = os.path.join(output_path, "stereotype_bias_data.jsonl")

        with open(prompts_path, "r") as f:
            for line in f:
                prompt: Dict = json.loads(line)["prompt"]
                prompt_text: str = prompt["text"]
                stereotype_topic_tag: str = prompt["stereotype_topic_tag"]
                demographic_group_tag: str = prompt["demographic_group_tag"]
                sys_prompt_type_tag: str = prompt["sys_prompt_type_tag"]

                # All instances contain a stereotype, so the model should ideally always disagree
                instance: Instance = Instance(
                    input=Input(text=prompt_text),
                    references=[
                        Reference(
                            Output(text=stereotype_topic_tag + " " + demographic_group_tag + " " + sys_prompt_type_tag),
                            tags=[stereotype_topic_tag, demographic_group_tag, sys_prompt_type_tag],
                        )
                    ],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances
