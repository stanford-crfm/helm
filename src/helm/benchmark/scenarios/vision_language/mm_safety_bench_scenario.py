import os
import json
from typing import Any, Dict, List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded


class MMSafetyBenchScenario(Scenario):
    """
    To evaluate the extent of this vulnerability in open-source VLMs, compiled a substantial dataset encompassing
    13 scenarios with a total of 5,040 text-image pairs

    @misc{liu2023queryrelevant,
          title={Query-Relevant Images Jailbreak Large Multi-Modal Models},
          author={Xin Liu and Yichen Zhu and Yunshi Lan and Chao Yang and Yu Qiao},
          year={2023},
          eprint={2311.17600},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

    Paper: https://arxiv.org/abs/2311.17600
    Website: https://isxinliu.github.io/Project/MM-SafetyBench/
    Questions: https://github.com/isXinLiu/MM-SafetyBench/tree/main/data/processed_questions
    """

    SUBSET_TO_DATASET_FOLDER: Dict[str, str] = {
        "illegal_activity": "01-Illegal_Activitiy",  # This is intentionally misspelled to match the original name
        "hate_speech": "02-HateSpeech",
        "malware_generation": "03-Malware_Generation",
        "physical_harm": "04-Physical_Harm",
        "economic_harm": "05-EconomicHarm",
        "fraud": "06-Fraud",
        "sex": "07-Sex",
        "political_lobbying": "08-Political_Lobbying",
        "privacy_violence": "09-Privacy_Violence",
        "legal_opinion": "10-Legal_Opinion",
        "financial_advice": "11-Financial_Advice",
        "health_consultation": "12-Health_Consultation",
        "government_decision": "13-Gov_Decision",
    }

    QUESTIONS_URL_TEMPLATE: str = (
        "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/{dataset}.json"
    )
    IMAGES_URL: str = "https://drive.google.com/uc?export=download&id=1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_"

    name = "mm_safety_bench"
    description = (
        "Expose the vulnerability of open-source VLMs with toxic and biased content "
        "([Liu et al., 2023](https://arxiv.org/abs/2311.17600))."
    )
    tags = ["vision-language", "bias", "toxicity"]

    def __init__(self, subset: str):
        super().__init__()
        assert subset in self.SUBSET_TO_DATASET_FOLDER, f"Invalid subset: {subset}"
        self._dataset: str = self.SUBSET_TO_DATASET_FOLDER[subset]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download all the images
        images_path: str = os.path.join(output_path, "MM-SafetyBench(imgs)")
        assert os.path.exists(images_path), (
            f"Images path does not exist: {images_path}. Download the images "
            f"from {self.IMAGES_URL}, unzip and place it at {output_path}"
        )
        # SD_TYPO seems to have the greatest attack success rate on the models they evaluated
        images_path = os.path.join(images_path, self._dataset, "SD_TYPO")
        assert os.path.exists(images_path)

        questions_path: str = os.path.join(output_path, f"{self._dataset}.json")
        questions_url: str = self.QUESTIONS_URL_TEMPLATE.format(dataset=self._dataset)
        ensure_file_downloaded(source_url=questions_url, target_path=questions_path)

        instances: List[Instance] = []

        with open(questions_path, "r") as questions_file:
            questions: Dict[str, Any] = json.load(questions_file)
            for question_id, question_data in questions.items():
                local_image_path: str = os.path.join(images_path, f"{question_id}.jpg")
                assert os.path.exists(local_image_path), f"Image does not exist: {local_image_path}"

                question: str = question_data["Rephrased Question"]
                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/jpeg"),
                    MediaObject(text=question, content_type="text/plain"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[],
                        split=TEST_SPLIT,
                    )
                )

        return instances
