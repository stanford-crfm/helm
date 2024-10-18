import json
import os
import os.path
import shutil
import datasets
from typing import List, Dict, Any, cast

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    ALL_SPLITS,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


def download_dataset(data_path: str, splits: List[str]):
    if False not in [os.path.exists(f"{data_path}/{split}.jsonl") for split in splits]:
        return

    # https://huggingface.co/docs/datasets/index
    ds_names: List[str] = datasets.list_datasets()
    # https://huggingface.co/casehold
    # https://huggingface.co/datasets/casehold/casehold
    ds_name = "casehold/casehold"
    if ds_name not in ds_names:
        raise Exception(f"{ds_name} not included in datasets")
    casehold: datasets.DatasetDict = cast(datasets.DatasetDict, datasets.load_dataset(ds_name))

    for split in splits:
        casehold[split].to_json(f"{data_path}/{split}.jsonl")

    # **WORK-AROUND**
    # since "test.jsonl" includes no label info., we use "validation.jsonl" as a substitute.
    if os.path.exists(f"{data_path}/test.jsonl"):
        os.remove(f"{data_path}/test.jsonl")
    shutil.copy(f"{data_path}/validation.jsonl", f"{data_path}/test.jsonl")


class CaseHOLDQAScenario(Scenario):
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

    """

    name = "casehold_qa"
    description = "CaseHOLD QA"
    tags = ["question_answering", "legal"]

    splits_dict = {TRAIN_SPLIT: "train", VALID_SPLIT: "validation", TEST_SPLIT: "test"}

    def __init__(self, splits: List[str] = ALL_SPLITS):
        super().__init__()
        self.splits = splits

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        download_dataset(data_path, list(self.splits_dict.values()))

        def to_instance(line: str, split: str) -> Instance:
            case: Dict[str, Any] = json.loads(line)
            example_id: int = case["example_id"]
            context: str = case["citing_prompt"]
            question: str = "holding statement"
            holdings: List[str] = [case[f"holding_{i}"] for i in range(5)]
            label: str = case["label"]
            instance: Instance = Instance(
                input=PassageQuestionInput(passage=context, question=question),
                references=[
                    Reference(Output(text=holdings[i]), tags=([CORRECT_TAG] if label == str(i) else []))
                    for i in range(5)
                ],
                split=split,
                id=str(example_id),
            )
            return instance

        instances: List[Instance] = []
        # TRAIN, VALID
        for split in self.splits:
            with open(f"{data_path}/{self.splits_dict[split]}.jsonl", mode="r") as f:
                for line in f.readlines():
                    instances.append(to_instance(line, split))

        return instances
