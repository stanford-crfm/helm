import os
import csv
import sys
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_file_downloaded

csv.field_size_limit(sys.maxsize)


class MedBulletsScenario(Scenario):
    """
    From "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions"
    (Chen et al.), MedBullet is a dataset comprising USMLE Step 2&3 style clinical questions. The dataset
    is designed to evaluate the performance of LLMs in answering and explaining challenging medical questions,
    emphasizing the need for explainable AI in medical QA.

    Example from the dataset:

    Question:
    A 42-year-old woman is enrolled in a randomized controlled trial to study cardiac function in the setting of
    several different drugs. She is started on verapamil and instructed to exercise at 50% of her VO2 max while
    several cardiac parameters are being measured. During this experiment, which of the following represents
    the relative conduction speed through the heart from fastest to slowest?

    A) AV node > ventricles > atria > Purkinje fibers
    B) Purkinje fibers > ventricles > atria > AV node
    C) Purkinje fibers > atria > ventricles > AV node
    D) Purkinje fibers > AV node > ventricles > atria

    Answer:
    The answer is C. Explanation: The conduction velocity of the structures of the heart is in the following order:
    Purkinje fibers > atria > ventricles > AV node. A calcium channel blocker such as verapamil would only slow
    conduction in the AV node.

    @Article{MedBullet,
    author = {Hanjie Chen and Zhouxiang Fang and Yash Singla and Mark Dredze},
    title = {Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions},
    year = {2023},
    abstract = {LLMs have demonstrated impressive performance in answering medical questions, such as passing scores
    on medical licensing examinations. However, medical board exam questions or general clinical questions do not
    capture the complexity of realistic clinical cases. Moreover, the lack of reference explanations means we cannot
    easily evaluate the reasoning of model decisions, a crucial component of supporting doctors in making complex
    medical decisions. To address these challenges, we construct two new datasets: JAMA Clinical Challenge and
    Medbullets. JAMA Clinical Challenge consists of questions based on challenging clinical cases, while Medbullets
    comprises USMLE Step 2&3 style clinical questions. Both datasets are structured as multiple-choice
    question-answering tasks, where each question is accompanied by an expert-written explanation. We evaluate four
    LLMs on the two datasets using various prompts. Experiments demonstrate that our datasets are harder than
    previous benchmarks. The inconsistency between automatic and human evaluations of model-generated explanations
    highlights the need to develop new metrics to support future research on explainable medical QA.}}

    Task:
    Given a clinical question with multiple-choice options, models must identify the correct answer and generate a
    response that includes the reasoning, as described in the expert-written explanation.
    """

    DATASET_DOWNLOAD_BASE_URL = (
        "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/refs/heads/main/medbullets/"
    )

    name = "medbullet"
    description = "A USMLE-style medical question dataset with multiple-choice answers and explanations."
    tags = ["reasoning", "biomedical"]

    # Define the possible answer choices
    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B", "C", "D", "E"]

    def __init__(self):
        super().__init__()
        # self.splits = {"_op4": TRAIN_SPLIT, "_op5": TEST_SPLIT}
        # limit to zero shot setting for now
        self.splits = {"_op5": TEST_SPLIT}

    def download_csv(self, output_path: str, split: str):
        """Download CSV files for the given split."""
        csv_path = os.path.join(output_path, f"medbullets{split}.csv")
        ensure_file_downloaded(
            source_url=f"{self.DATASET_DOWNLOAD_BASE_URL}/medbullets{split}.csv",
            target_path=csv_path,
            unpack=False,
        )
        return csv_path

    def process_csv(self, csv_path: str, split: str) -> List[Instance]:
        """Read and process a CSV file to generate instances."""
        instances: List[Instance] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Validate required fields
                if not row.get("question") or not row.get("answer_idx") or not row.get("opa"):
                    print(f"Skipping invalid row: {row}")
                    continue

                # Map answers to indices
                option_map = {
                    "A": row.get("opa", "Not applicable"),
                    "B": row.get("opb", "Not applicable"),
                    "C": row.get("opc", "Not applicable"),
                    "D": row.get("opd", "Not applicable"),
                    "E": row.get("ope", "Not applicable"),
                }

                # Correct answer text
                correct_option = row["answer_idx"]

                # Build references using POSSIBLE_ANSWER_CHOICES
                references = [
                    Reference(
                        Output(text=option_map.get(option, "Not applicable")),
                        tags=[CORRECT_TAG] if option == correct_option else [],
                    )
                    for option in self.POSSIBLE_ANSWER_CHOICES
                ]

                # Create instance
                instance = Instance(
                    input=Input(text=row["question"]),
                    references=references,
                    split=split,
                )
                instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Download and process dataset to generate instances."""
        instances: List[Instance] = []
        for split_suffix, split in self.splits.items():
            csv_path = self.download_csv(output_path, split_suffix)
            instances.extend(self.process_csv(csv_path, split))
        return instances
