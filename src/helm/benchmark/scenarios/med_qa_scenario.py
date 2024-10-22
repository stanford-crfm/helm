import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    Input,
    Output,
)


class MedQAScenario(Scenario):
    """
    From "What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams"
    (Jin et al.), MedQA is an open domain question answering dataset composed of questions from professional medical
    board exams.

    From Jin et al., "to comply with fair use of law ,we shuffle the order of answer options and randomly delete
    one of the wrong options for each question for USMLE and MCMLE datasets, which results in four options with one
    right option and three wrong options".
    We use the 4-options, English subset ("US") of the dataset, which contains 12,723 questions.

    The following is an example from the dataset:

    {
      "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states
      it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She
      otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C),
      blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air.
      Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the
      following is the best treatment for this patient?",
      "answer": "Nitrofurantoin",
      "options": {
        "A": "Ampicillin",
        "B": "Ceftriaxone",
        "C": "Ciprofloxacin",
        "D": "Doxycycline",
        "E": "Nitrofurantoin"
      },
      "meta_info": "step2&3",
      "answer_idx": "E"
    }

    Paper: https://arxiv.org/abs/2009.13081
    Code: https://github.com/jind11/MedQA

    @article{jin2020disease,
      title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset
             from Medical Exams},
      author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
      journal={arXiv preprint arXiv:2009.13081},
      year={2020}
    }
    """

    DATASET_DOWNLOAD_URL: str = (
        "https://drive.google.com/uc?export=download&id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw&confirm=t"
    )

    name = "med_qa"
    description = "An open domain question answering (QA) dataset collected from professional medical board exams."
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            split_file_name: str = f"phrases_no_exclude_{'dev' if split == VALID_SPLIT else split}.jsonl"
            split_path: str = os.path.join(data_path, "data_clean", "questions", "US", "4_options", split_file_name)

            with open(split_path, "r") as f:
                for line in f:
                    example: Dict = json.loads(line.rstrip())
                    assert len(example["options"]) == 4, f"Expected 4 possible answer choices: {example['options']}"
                    references: List[Reference] = [
                        # Some answers have extraneous characters e.g., 'Pulmonary embolism\n"'.
                        # Filed an issue: https://github.com/jind11/MedQA/issues/5
                        # Value of "answer_idx" corresponds to the letter of the correct answer.
                        Reference(
                            Output(text=answer.rstrip('"').rstrip()),
                            tags=[CORRECT_TAG] if index == example["answer_idx"] else [],
                        )
                        for index, answer in example["options"].items()
                    ]
                    instance: Instance = Instance(
                        input=Input(text=example["question"]),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)

        return instances
