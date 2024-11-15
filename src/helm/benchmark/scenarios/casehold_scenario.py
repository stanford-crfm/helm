from typing import List
import os
import os.path

from datasets import load_dataset, DatasetDict

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Output,
)


class CaseHOLDScenario(Scenario):
    """
     CaseHOLD QA
       CaseHOLD is a multiple choice question answering task derived from legal citations in judicial rulings.
       CaseHOLD consists of ~53,000 questions, mined from the Harvard Law Library case law corpus.

     Dataset repository
       https://huggingface.co/datasets/casehold/casehold
     Publication
       "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset"
       ICAIL, 2021
       https://reglab.stanford.edu/data/casehold-benchmark/
       https://arxiv.org/abs/2104.08671

    Data content
      The citing context from the judicial decision serves as the prompt for the question.
      The answer choices are holding statements derived from citations following text in a legal decision.
      There are five answer choices for each citing text.
      The correct answer is the holding statement that corresponds to the citing text.
      The four incorrect answers are other holding statements.

    """  # noqa: E501

    name = "casehold"
    description = "CaseHOLD (Case Holdings On Legal Decisions) is a multiple choice question answering scenario where the task is to identify the relevant holding of a cited case [(Zheng et al, 2021)](https://arxiv.org/pdf/2104.08671.pdf)."  # noqa: E501
    tags = ["question_answering", "legal"]

    # Note: Skip the validation split since we don't need it
    HELM_SPLIT_NAME_TO_DATASETS_SPLIT_NAME = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}
    NUM_REFERENCES = 5

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(
            "casehold/casehold",
            "all",
            cache_dir=data_path,
        )

        instances: List[Instance] = []
        for helm_split_name, datasets_split_name in self.HELM_SPLIT_NAME_TO_DATASETS_SPLIT_NAME.items():
            split_data = dataset[datasets_split_name]
            for example in split_data:
                example_id = example["example_id"]
                citing_prompt = example["citing_prompt"]
                holdings = [example[f"holding_{i}"] for i in range(self.NUM_REFERENCES)]
                correct_label: str = example["label"]
                references = [
                    Reference(Output(text=holdings[i]), tags=([CORRECT_TAG] if correct_label == str(i) else []))
                    for i in range(self.NUM_REFERENCES)
                ]
                instance = Instance(
                    input=Input(text=citing_prompt),
                    references=references,
                    split=helm_split_name,
                    id=f"id{example_id}",
                )
                instances.append(instance)
        return instances
