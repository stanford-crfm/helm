import json
import logging
import os
from typing import Dict, List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_file_downloaded


class MedBulletsScenario(Scenario):
    """
    From "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions"
    (Chen et al.), MedBullets is a dataset comprising USMLE Step 2&3 style clinical questions. The dataset
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

    @article{chen2024benchmarking,
        title={Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions},
        author={Chen, Hanjie and Fang, Zhouxiang and Singla, Yash and Dredze, Mark},
        journal={arXiv preprint arXiv:2402.18060},
        year={2024}
    }

    Task:
    Given a clinical question with multiple-choice options, models must identify the correct answer and generate a
    response that includes the reasoning, as described in the expert-written explanation.
    """

    DATASET_DOWNLOAD_BASE_URL: str = "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/refs/heads/main/medbullets/"
    DATASET_JSON_NAMES: List[str] = ["medbullets_op4.json", "medbullets_op5.json"]

    name = "medbullets"
    description = "USMLE Step 2&3 style clinical questions with explanations."
    tags = ["reasoning", "biomedical", "medicine"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # The dataset has no official train/test split.
        helm_split_name = TEST_SPLIT

        instances: List[Instance] = []
        for json_name in self.DATASET_JSON_NAMES:
            json_path = os.path.join(output_path, json_name)
            ensure_file_downloaded(
                source_url=f"{self.DATASET_DOWNLOAD_BASE_URL}/{json_name}",
                target_path=json_path,
                unpack=False,
            )

            # Load the JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                data: List[Dict] = json.load(f)

                question_ids = list(data["question"].keys())
                for question_id in question_ids:
                    references = self._get_references(data, question_id)

                    if not references:
                        continue

                    instances.append(
                        Instance(
                            input=Input(text=data["question"][question_id]),
                            references=references,
                            split=helm_split_name,
                        )
                    )

        return instances

    def _get_references(self, data: Dict, question_id: str) -> List[Reference]:
        """Get references for a given question from the dataset."""
        answer = data["answer"][question_id]

        options = []
        try:
            # Map from answer_idx to the corresponding option text
            options.extend(
                [
                    data["opa"][question_id],
                    data["opb"][question_id],
                    data["opc"][question_id],
                    data["opd"][question_id],
                ]
            )
        except KeyError:
            logging.error(
                f"Failed to process references for MedBullets question {question_id}"
            )
            return []

        # MedBullets includes a dataset with 4 options and another with 5 options
        if "ope" in data:
            options.append(data["ope"][question_id])

        # XXX: Should we verify we have one correct answer?
        return [
            Reference(
                Output(text=option),
                tags=[CORRECT_TAG] if option == answer else [],
            )
            for option in options
        ]


class MedBulletsFreeTextScenario(MedBulletsScenario):
    """MedBullets free-text scenario.

    An adaptation of the MedBullets dataset that uses the explanation as the answer.
    """

    name = "medbullets-freetext"

    def _get_references(self, data: Dict, question_id: str) -> List[Reference]:
        """Get references for a given question from the dataset."""
        answer = data["answer"][question_id]
        explanation = data["explanation"][question_id].split("Incorrect Answers:")[0]
        return [
            Reference(
                Output(text=f"{answer}. {explanation}"),
                tags=[CORRECT_TAG],
            )
        ]
