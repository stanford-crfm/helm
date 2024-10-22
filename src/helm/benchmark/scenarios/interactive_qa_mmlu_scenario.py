import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Instance, TRAIN_SPLIT, TEST_SPLIT
from helm.benchmark.scenarios.mmlu_scenario import MMLUScenario


class InteractiveQAMMLUScenario(MMLUScenario):
    """
    The Massive Multitask Language Understanding benchmark from this paper
    https://arxiv.org/pdf/2009.03300.pdf

    For InteractiveQA, we used a small subset of the original test set.
    """

    name = "interactive_qa_mmlu"
    description = "Massive Multitask Language Understanding (InteractiveQA subset)"
    tags = ["knowledge", "multiple_choice"]

    VALID_SUBJECTS: List[str] = [
        "college_chemistry",
        "global_facts",
        "miscellaneous",
        "nutrition",
        "us_foreign_policy",
    ]
    INTERACTIVE_QA_DIR: str = "interactive_qa"
    MMLU_SPLITS: List[str] = ["dev", "test"]

    def __init__(self, subject: str):
        assert subject in InteractiveQAMMLUScenario.VALID_SUBJECTS, f"Invalid subject for Interactive QA: {subject}"
        super().__init__(subject)

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the MMLU data and the subsets for InteractiveQA
        data_path: str = os.path.join(output_path, "data")
        self.download_mmlu(data_path)

        test_path: str = os.path.join(data_path, "test", InteractiveQAMMLUScenario.INTERACTIVE_QA_DIR)
        ensure_file_downloaded(
            source_url="https://worksheets.codalab.org/rest/bundles/0x4d49146fe16c4559bc64af6cbc04992d/"
            "contents/blob/",
            target_path=test_path,
            unpack=True,
            unpack_type="untar",
        )

        # Read all the instances
        instances: List[Instance] = []
        for mmlu_split in InteractiveQAMMLUScenario.MMLU_SPLITS:
            if mmlu_split == "test":
                split = TEST_SPLIT
                split_path = test_path
            else:
                split = TRAIN_SPLIT
                split_path = os.path.join(data_path, mmlu_split)

            csv_path: str = os.path.join(split_path, f"{self.subject}_{mmlu_split}.csv")
            instances.extend(self.process_csv(csv_path, split))

        return instances
