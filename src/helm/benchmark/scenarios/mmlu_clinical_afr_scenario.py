import csv
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class MMLU_Clinical_Afr_Scenario(Scenario):
    """
    https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages
    """

    name = "mmlu_clinical_afr"
    description = "Massive Multitask Language Understanding (MMLU) translated into 11 African low-resource languages"
    tags = ["knowledge", "multiple_choice", "low_resource_languages"]

    def __init__(self, subject: str = "clinical_knowledge", lang: str = "af"):
        super().__init__()
        self.subject: str = subject
        self.lang: str = lang

    def download_mmlu_clinical_afr(self, path: str):
        ensure_file_downloaded(
            source_url="https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages/raw/refs/heads/main/data/evaluation_benchmarks_afr_release.zip",  # noqa: E501
            target_path=path,
            unpack=True,
            unpack_type="unzip",
        )

    def process_csv(self, csv_path: str, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Reading {csv_path}")
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:

                question, answers, correct_choice = row[0], row[1:-1], row[-1]
                answers_dict = dict(zip(["A", "B", "C", "D"], answers))
                correct_answer: str = answers_dict[correct_choice]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=split,
                )
                instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        desired_dir = "mmlu_cm_ck_vir"
        data_path: str = os.path.join(output_path, desired_dir)
        self.download_mmlu_clinical_afr(data_path)

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "dev": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            csv_path: str = os.path.join(data_path, desired_dir, f"{self.subject}_{split}_{self.lang}.csv")
            if not os.path.exists(csv_path):
                hlog(f"{csv_path} doesn't exist, skipping")
                continue
            instances.extend(self.process_csv(csv_path, splits[split]))

        return instances
