import json
import os
import random
from typing import List, Dict

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, TEST_TAG

TOXIC_TAG = "toxic"
NONTOXIC_TAG = "non-toxic"

class BOLDScenario(Scenario):
    """
    The BOLD dataset is from the paper:

        https://www.amazon.science/publications/bold-dataset-and-metrics-for-measuring-biases-in-open-ended-language-generation

    The dataset contains 23679 prompts across five domains.
    """

    name = "bold"
    description = "a large-scale dataset that consists of 23,679 English text generation prompts for bias benchmarking across five domains: profession, gender, race, religion, and political ideology."
    tags = ["harms", "bias"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        categories = [
            "gender",
            "political_ideology",
            "profession",
            "race", 
            "religious_ideology",
        ]
        for category in categories:
            ensure_file_downloaded(
                source_url=f"https://raw.githubusercontent.com/amazon-research/bold/main/prompts/{category}_prompt.json",
                target_path=data_path,
                unpack=False,
            )

        instances: List[Instance] = []
        prompts_path: str = data_path
        with open(prompts_path, "r") as f:
            for line in f:
#                print(f"FIRST LINE: {json.loads(line)}")
                prompt_topics_dict: Dict = json.loads(line)
                
                for prompt_topic in prompt_topics_dict.values():
                    for prompt_text in prompt_topic.values():
                        prompt_text = prompt_text[0] #stored as a list containing a single string
                        tags: List[str] = [
                            TEST_TAG,
                            NONTOXIC_TAG, # TODO: do we want to assess a prompt's toxicity?
                        ]

                        instance: Instance = Instance(input=prompt_text, references=[], tags=tags)
                        instances.append(instance)

        random.seed(0)
        random.shuffle(instances)

        return instances
