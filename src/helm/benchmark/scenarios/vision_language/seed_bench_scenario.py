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


class SEEDBenchScenario(Scenario):
    """
    SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension

    Based on powerful Large Language Models (LLMs), recent generative Multimodal
    Large Language Models (MLLMs) have gained prominence as a pivotal research area.
    In Seed-Bench, we address the evaluation of generative comprehension in MLLMs
    as a preliminary step towards a comprehensive assessment of generative models.
    SEED-Bench consists of 19K multiple choice questions with accurate human annotations
    (x 6 larger than existing benchmarks), which spans 12 evaluation dimensions
    including the comprehension of both the image and video modality. We select 9
    evaluation aspects that take image as the input. In the benchmark,
    Multiple-choice questions with groundtruth options derived from human
    annotation enables an objective and efficient assessment of model performance,
    eliminating the need for human or GPT intervention during evaluation. We employ
    the multiple-choice metric for evaluating the performance of models.

    @article{li2023seed,
        title={Seed-bench: Benchmarking multimodal llms with generative comprehension},
        author={Li, Bohao and Wang, Rui and Wang, Guangzhi and Ge, Yuying and Ge, Yixiao and Shan, Ying},
        journal={arXiv preprint arXiv:2307.16125},
        year={2023}
    }

    Paper: https://arxiv.org/abs/2307.16125
    """

    SEED_BENCH_HUGGINGFACE_DATASET_NAME: str = "lmms-lab/SEED-Bench"

    SUBJECTS: Dict[str, int] = {
        "scene-understanding": 1,
        "instance-identity": 2,
        "instance-attributes": 3,
        "instance-location": 4,
        "instances-counting": 5,
        "spatial-relation": 6,
        "instance-interaction": 7,
        "visual-reasoning": 8,
        "text-understanding": 9,
    }

    name = "seed_bench"
    description = (
        "Evaluate multimodal models on 9 evaluation aspects " "([Li et al., 2023](https://arxiv.org/abs/2307.16125))."
    )
    tags = ["vision-language"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

    def get_subject_name(self, subject_name: str) -> str:
        return "-".join(subject_name.lower().split())

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
                self.SEED_BENCH_HUGGINGFACE_DATASET_NAME,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            question_type_key: str = self.get_subject_name(self._subject)
            if row["question_type_id"] != self.SUBJECTS[question_type_key]:
                continue
            question_id: str = row["question_id"]
            # Download the image
            # Save the image locally
            image_path: str = os.path.join(images_path, f"{question_id}.png")
            if not os.path.exists(image_path):
                # some images are CMYK mode, convert to RGB.
                row["image"][0].convert("RGB").save(image_path, "PNG", optimize=True)

            # Add the references
            references: List[Reference] = []
            question: str = row["question"]
            answer: str
            content: List[MediaObject]
            options: List[str] = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
            answer = row["answer"].strip()
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
