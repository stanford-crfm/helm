import json
import os
import random
from typing import List

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG


AMBIGUOUS_TAG = "ambiguous"
NON_AMBIGUOUS_TAG = "non-ambiguous"
NEGATIVE_TAG = "negative"
NON_NEGATIVE_TAG = "non-negative"


class BBQScenario(Scenario):
    """
    The BBQ dataset is from the paper:

        https://arxiv.org/pdf/2110.08193.pdf

    The dataset contains TODO prompts, TODO in ambiguous contexts and TODO in non-ambiguous contexts.
    """

    name = "bbq"
    description = "Bias analysis of LM question-answering."
    tags = ["harms", "bias"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")

        bbq_categories = [
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Race_x_SES",
            "Race_x_gender",
            "Religion",
            "SES",
            "Sexual_orientation",
        ]

        for bbq_category in bbq_categories:

            ensure_file_downloaded(
                source_url=f"https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{bbq_category}.jsonl",
                target_path=data_path,
                unpack=False,
            )

        instances: List[Instance] = []
        prompts_path: str = data_path
            
        with open(prompts_path, "r") as f:
            for line in f:

                loaded_line = json.loads(line)
                is_negative: bool = (loaded_line["question_polarity"] == "neg")
                is_ambiguous: bool = (loaded_line["context_condition"] == "ambig")
                context: str = loaded_line["context"]
                question: str = loaded_line["question"]
                ans0: str = loaded_line["ans0"]
                ans1: str = loaded_line["ans1"]
                ans2: str = loaded_line["ans2"]
                label = loaded_line["label"]
                answers = [ans0, ans1, ans2]

                tags: List[str] = [
                    NEGATIVE_TAG if is_negative else NON_NEGATIVE_TAG,
                    AMBIGUOUS_TAG if is_ambiguous else NON_AMBIGUOUS_TAG,
                ]

                def answer_to_reference(answer):
                    if answer == label:
                        tags.append(CORRECT_TAG)
                    return Reference(output=answer, tags=tags)

                instance: Instance = Instance(
                    input=f"{context} {question}", references=list(map(answer_to_reference, answers)), tags=tags
                )
                instances.append(instance)

        random.seed(0)
        random.shuffle(instances)
        
        print(f"Instaces is of length: {len(instances)}")

        return instances
