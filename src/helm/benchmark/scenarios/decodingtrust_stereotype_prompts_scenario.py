import json
import os
import pandas as pd
from typing import List, Dict, Optional
from helm.benchmark.scenarios.scenario import Instance

from helm.common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, TEST_SPLIT, Input


class DecodingTrustStereotypePromptsScenario(Scenario):
    """
    The DecodingTrustStereotypePrompts dataset is from the paper:
    https://arxiv.org/abs//2306.11698

    The dataset contains 1,152 manually crafted stereotype user prompts.
    The prompts cover 16 stereotype topics (for e.g., drug addiction, HIV, etc.),
    and 24 demographic groups spanning 7 demographic factors: race/ethnicity (Asians, Black
    people, etc.), gender/sexual orientation (homosexuals, men, and women),
    nationality (Mexicans, Americans, etc.), age (old and young people), religion (Muslims,
    Jews, etc.), disability (physically disabled and able-bodied people), and socioeconomic status (poor
    and rich people).
    """

    name = "decodingtrust_stereotype_prompts"
    description = (
        "Stereotype bias analysis of LM generations when fed manually crafted user prompts that contains stereotypes."
    )
    tags = ["harms", "stereotype", "bias"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="",  # to be filled
            target_path=data_path,
            unpack=True,
        )

        instances: List[Instance] = []
        prompts_path: str = os.path.join(data_path, "prompts.jsonl")

        with open(prompts_path, "r") as f:
            for line in f:
                prompt: Dict = json.loads(line)["prompt"]
                prompt_text: str = prompt["text"]

                # All instances contain a stereotype, so the model should ideally always disagree
                instance: Instance = Instance(input=Input(text=prompt_text), references=[])
                instances.append(instance)

        return instances
