import os.path
from typing import List

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


class MMEScenario(Scenario):
    """
    MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models

    Multimodal Large Language Model (MLLM) relies on the powerful LLM to perform
    multimodal tasks, showing amazing emergent abilities in recent studies. However,
    it is difficult for these case studies to fully reflect the performance of MLLM,
    lacking a comprehensive evaluation. In MME, we fill in this blank, presenting
    the first comprehensive MLLM Evaluation benchmark MME. It measures both perception
    and cognition abilities on a total of 14 subtasks. In order to avoid data leakage
    that may arise from direct use of public datasets for evaluation, the annotations
    of instruction-answer pairs are all manually designed. The concise instruction design
    allows us to fairly compare MLLMs, instead of struggling in prompt engineering.
    Besides, with such an instruction, we can also easily carry out quantitative
    statistics. We rephrase the answer type of MME to multiple-choice question-answering.
    We use the multiple-choice metrics for 14 different evaluation tasks.

    @article{fu2023mme,
        title={MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models},
        author={Fu, Chaoyou and Chen, Peixian and Shen, Yunhang and Qin, Yulei and
        Zhang, Mengdan and Lin, Xu and Yang, Jinrui and Zheng, Xiawu and Li, Ke and
        Sun, Xing and Wu, Yunsheng and Ji, Rongrong},
        journal={arXiv preprint arXiv:2306.13394},
        year={2023}
    }

    Paper: https://arxiv.org/abs/2306.13394
    """

    MME_HUGGINGFACE_DATASET_NAME: str = "lmms-lab/MME"

    SUBJECTS: List[str] = [
        "existence",
        "scene",
        "posters",
        "color",
        "OCR",
        "position",
        "celebrity",
        "artwork",
        "commonsense_reasoning",
        "numerical_calculation",
        "landmark",
        "count",
        "text_translation",
        "code_reasoning",
    ]

    name = "mme"
    description = (
        "Evaluate multimodal models on their perception and cognition abilities on a total of 14 subtasks "
        "([Fu et al., 2023](https://arxiv.org/abs/2306.13394))."
    )
    tags = ["vision-language"]
    options: List[str] = ["Yes", "No"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

    def get_label_from_answer(self, answer: str):
        label: str
        if answer == "Yes":
            label = "A"
        elif answer == "No":
            label = "B"
        else:
            raise NotImplementedError(f"Invalid answer: {answer}")
        return label

    def remove_question_suffix_for_mcqa(self, question: str):
        return question.replace("Please answer yes or no.", "").strip()

    def get_question_id(self, question_id: str):
        return question_id.split(".")[0].replace("/", "-")

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        # There is only the test split in Unicorn benchmark
        instances: List[Instance] = []
        # Process the test set
        # Two open-ended generation instances and
        # one multi-choice generation instance per row
        for row in tqdm(
            load_dataset(
                self.MME_HUGGINGFACE_DATASET_NAME,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            if row["category"] != self._subject:
                continue
            question_id: str = self.get_question_id(row["question_id"])
            # Save the image locally
            image_path: str = os.path.join(images_path, f"{question_id}.png")
            if not os.path.exists(image_path):
                row["image"].save(image_path)

            question: str = self.remove_question_suffix_for_mcqa(row["question"])
            answer: str = row["answer"]
            references: List[Reference] = []

            answer = self.get_label_from_answer(answer)
            # The given correct answer is a letter, but we need an index
            correct_answer_index: int = ord(answer) - ord("A")
            # The options are originally appended to the question

            for i, option in enumerate(self.options):
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
