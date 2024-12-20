import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)


class GSM8KScenario(Scenario):
    """Task from "Training Verifiers to Solve Math Word Problems" (Cobbe et al. 2021): https://arxiv.org/abs/2110.14168

    Evaluates the capacity of a model to solve grade school math problems, when prompted to include reasoning.
    Encourages the model to work through the problem in a step-by-step way.

    Example from dataset (line breaks added for readability):

    ```
    "question":
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
        How many clips did Natalia sell altogether in April and May?",
    "answer":
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n
        #### 72"
    ```

    Also, incorporates prompting methods from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"
    (Wei et al. 2021): https://arxiv.org/abs/2201.11903

    For example, we use "The answer is" before the answer, and remove line breaks within the answer.
    """

    name = "gsm"
    description = "Grade school math dataset with 8.5K examples (GSM8K)."
    tags = ["reasoning", "math"]

    def get_instances(self, output_path: str) -> List[Instance]:
        splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        base_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
        instances: List[Instance] = []

        for split, split_tag in splits.items():  # Iterate over the splits
            source_url: str = f"{base_url}/{split}.jsonl"
            data_path: str = os.path.join(output_path, f"gsm_data_{split}")
            ensure_file_downloaded(source_url=source_url, target_path=data_path)

            with open(data_path, "r") as f:
                for line in f.readlines():
                    example: Dict = json.loads(line)
                    answer: str = example["answer"].replace("####", "The answer is").replace("\n", " ") + "."
                    instances.append(
                        Instance(
                            input=Input(text=example["question"]),
                            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                            split=split_tag,  # Must assign split tag to instance.
                        ),
                    )
        return instances
