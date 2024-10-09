import os
import json
from typing import Any, Dict, List

from helm.benchmark.scenarios.scenario import (
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded


class GQAScenario(Scenario):
    """
    Questions about real-world visual reasoning and compositional QA

    @misc{hudson2019gqa,
          title={GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering},
          author={Drew A. Hudson and Christopher D. Manning},
          year={2019},
          eprint={1902.09506},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }

    Paper: https://arxiv.org/abs/1902.09506
    Website: https://cs.stanford.edu/people/dorarad/gqa/about.html
    """

    QUESTIONS_URL: str = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    IMAGES_URL: str = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

    name = "gqa"
    description = (
        "Questions about real-world visual reasoning and compositional QA "
        "([Hudson and Manning, 2019](https://arxiv.org/abs/1902.09506))."
    )
    tags = ["vision-language", "reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        questions_path: str = os.path.join(output_path, "questions")
        ensure_file_downloaded(
            source_url=self.QUESTIONS_URL, target_path=questions_path, unpack=True, unpack_type="unzip"
        )

        images_path: str = os.path.join(output_path, "images")
        ensure_file_downloaded(source_url=self.IMAGES_URL, target_path=images_path, unpack=True, unpack_type="unzip")

        instances: List[Instance] = []
        for helm_split in ALL_SPLITS:
            if helm_split == TEST_SPLIT:
                # The test split doesn't have annotations
                continue

            split: str = "val" if helm_split == VALID_SPLIT else helm_split

            # Read the questions from the JSON
            questions_split_path: str = os.path.join(questions_path, f"{split}_balanced_questions.json")
            with open(questions_split_path, "r") as questions_file:
                questions: Dict[str, Any] = json.load(questions_file)
                for question_id, question_data in questions.items():
                    question: str = question_data["question"]
                    short_answer: str = question_data["answer"]
                    full_answer: str = question_data["fullAnswer"]

                    image_id: str = question_data["imageId"]
                    local_image_path: str = os.path.join(images_path, f"{image_id}.jpg")

                    content: List[MediaObject] = [
                        MediaObject(text=question, content_type="text/plain"),
                        MediaObject(location=local_image_path, content_type="image/jpeg"),
                    ]
                    instances.append(
                        Instance(
                            Input(multimedia_content=MultimediaObject(content)),
                            references=[
                                Reference(Output(text=short_answer), tags=[CORRECT_TAG]),
                                Reference(Output(text=full_answer), tags=[CORRECT_TAG]),
                            ],
                            split=helm_split,
                        )
                    )

        return instances
