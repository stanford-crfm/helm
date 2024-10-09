import os.path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class MultipanelVQAScenario(Scenario):
    """
    Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA

    We introduce Multipanel Visual Question Answering (MultipanelVQA), a novel benchmark
    comprising 6,600 triplets of questions, answers, and multipanel images that specifically
    challenge models in comprehending multipanel images. Our evaluation shows that questions in
    the MultipanelVQA benchmark pose significant challenges to the state-of-the-art Large Vision
    Language Models (LVLMs) tested, even though humans can attain approximately 99% accuracy on
    these questions. There are two types of questions in two different situations in the
    MultipanelVQA benchmark: multiple-choice or open-ended generation paired with real-world or
    synthetic images. We use the multiple-choice metrics and the exact match metric for two
    different question-answering types, respectively.

    @article{fan2024muffin,
    title={Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA},
    author={Fan, Yue and Gu, Jing and Zhou, Kaiwen and Yan, Qianqi and Jiang, Shan and
    Kuo, Ching-Chen and Guan, Xinze and Wang, Xin Eric},
    journal={arXiv preprint arXiv:2401.15847},
    year={2024}
    }

    Paper: https://arxiv.org/abs/2401.15847
    """

    MULTIPANELVQA_HUGGINGFACE_DATASET_NAME: Dict[str, str] = {
        "synthetic": "yfan1997/MultipanelVQA_synthetic",
        "real-world": "yfan1997/MultipanelVQA_real-world",
    }

    SUBJECTS: List[str] = ["synthetic", "real-world"]

    name = "multipanelvqa"
    description = "Evaluate multimodal models on  ([paper](https://arxiv.org/abs/2401.15847))."
    tags = ["vision-language"]

    def __init__(self, subject: str, question_type: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

        assert question_type in ["multiple-choice", "open"], f"Invalid question type: {question_type}"
        self._question_type: str = question_type

    def convert_text_answer_to_option(self, text_answer: str, question: str):
        option_answer: str
        # Some answer may have a ')' with it
        if len(text_answer) <= 3:
            option_answer = text_answer[0]
        else:
            # There are examples where the answer is the text answer
            # instead of an option
            for line in question.split("\n"):
                if text_answer in line:
                    option_answer = line[0]
                    break
        return option_answer.upper()

    def split_options_and_question(self, original_question: str):
        question_and_options: List[str] = [item.strip().lower() for item in original_question.split("\n")]
        last_append_phrase: str = "(please select one)"
        question: str = question_and_options[0]
        options: List[str] = []
        if len(question_and_options) >= 6:
            for item in question_and_options[1:]:
                if last_append_phrase in item:
                    break
                options.append(item[3:])
        elif len(question_and_options) == 5:
            for item in question_and_options[1:]:
                if last_append_phrase in item:
                    item = item[: -len(last_append_phrase)]
                options.append(item[3:])
        return question, options

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        # There is only the test split in Unicorn benchmark
        instances: List[Instance] = []
        # Process the test set
        # Two open-ended generation instances and
        # one multi-choice generation instance per row
        for image_index, row in enumerate(
            tqdm(
                load_dataset(
                    self.MULTIPANELVQA_HUGGINGFACE_DATASET_NAME[self._subject],
                    split=TEST_SPLIT,
                    cache_dir=output_path,
                )
            )
        ):
            # Download the image
            # Save the image locally
            image_path: str = os.path.join(images_path, f"{image_index}.png")
            if not os.path.exists(image_path):
                row["image"].save(image_path)

            # Add the references
            references: List[Reference] = []
            question: str
            answer: str
            content: List[MediaObject]
            if self._question_type == "open":
                question_1: str = row["question_1"]
                question_2: str = row["question_2"]
                answer_1: str = row["answer_1"]
                answer_2: str = row["answer_2"]
                for answer, question in zip([answer_1, answer_2], [question_1, question_2]):
                    content = [
                        MediaObject(location=image_path, content_type="image/png"),
                        MediaObject(text=question, content_type="text/plain"),
                    ]
                    instances.append(
                        Instance(
                            Input(multimedia_content=MultimediaObject(content)),
                            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                            split=TEST_SPLIT,
                        )
                    )
            else:
                options: List[str]
                original_question: str = row["question_3"]
                question, options = self.split_options_and_question(original_question)
                answer = row["answer_3"].strip()
                answer = self.convert_text_answer_to_option(answer, original_question)
                # The given correct answer is a letter, but we need an index
                correct_answer_index: int = ord(answer) - ord("A")
                # The options are originally appended to the question

                for i, option in enumerate(options):
                    reference: Reference
                    is_correct: bool = i == correct_answer_index
                    reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                    references.append(reference)

                content = [
                    MediaObject(location=image_path, content_type="image/png"),
                    MediaObject(text=question, content_type="text/plain"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=references,
                        split=TEST_SPLIT,
                    )
                )

        return instances
