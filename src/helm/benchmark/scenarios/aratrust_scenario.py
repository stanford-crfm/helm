import os
from typing import List

import datasets

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class AraTrustScenario(Scenario):
    """AraTrust: An Evaluation of Trustworthiness for LLMs in Arabic

    EXPERIMENTAL: This scenario may have future reverse incompatible changes.

    AraTrust is a comprehensive Trustworthiness benchmark for LLMs in Arabic.
    AraTrust comprises 522 human-written multiple-choice questions addressing
    diverse dimensions related to truthfulness, ethics, safety, physical health,
    mental health, unfairness, illegal activities, privacy, and offensive language.

    - https://huggingface.co/datasets/asas-ai/AraTrust
    - https://arxiv.org/abs/2403.09017

    Citation:

    ```
    @misc{alghamdi2024aratrustevaluationtrustworthinessllms,
      title={AraTrust: An Evaluation of Trustworthiness for LLMs in Arabic},
      author={Emad A. Alghamdi and Reem I. Masoud and Deema Alnuhait and Afnan Y. Alomairi and Ahmed Ashraf and Mohamed Zaytoon},
      year={2024},
      eprint={2403.09017},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.09017},
    }
    ```
    """  # noqa: E501

    name = "aratrust"
    description = "aratrust"
    tags = ["trustworthiness"]

    CATEGORIES = [
        "Ethics",
        "Illegal",
        "Mental Health",
        "Offensive",
        "Physical Health",
        "Privacy",
        "Trustfulness",
        "Unfairness",
    ]
    OPTION_KEYS = ["A", "B", "C"]

    def __init__(self, category: str):
        super().__init__()
        category = category.replace("_", " ")
        if category not in self.CATEGORIES and category != "all":
            raise Exception(f"Unknown category {category}")
        self.category = category

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset: datasets.Dataset = datasets.load_dataset(
            "asas-ai/AraTrust",
            revision="d4dd124ed5b90aeb65a7dda7d88e34fb464a31ec",
            cache_dir=cache_dir,
            split="test",
        )
        instances: List[Instance] = []
        for row_index, row in enumerate(dataset):
            if self.category != "all" and self.category != row["Category"]:
                continue
            question_text = row["Question"]
            option_texts = [row[option_key] for option_key in self.OPTION_KEYS if row[option_key]]
            joined_option_texts = "\n".join(option_texts)
            input = Input(text=f"{question_text}\n\n{joined_option_texts}\n")
            references = [Reference(output=Output(text=row["Answer"]), tags=[CORRECT_TAG])]
            instance = Instance(
                id=f"id{row_index}",
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
