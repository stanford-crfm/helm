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


class VibeEvalScenario(Scenario):
    """
    Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models

    We introduce Vibe-Eval: a new open benchmark and framework for evaluating multimodal chat
    models. Vibe-Eval consists of 269 visual understanding prompts, including 100 of hard
    difficulty, complete with gold-standard responses authored by experts. Vibe-Eval is
    open-ended and challenging with dual objectives: (i) vibe checking multimodal chat models
    for day-to-day tasks and (ii) rigorously testing and probing the capabilities of present
    frontier models. Notably, our hard set contains >50% questions that all frontier models
    answer incorrectly. We also discuss trade-offs between human and automatic evaluation,
    and show that automatic model evaluation using Reka Core roughly correlates to human judgment.

    @article{padlewski2024vibe,
    title={Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models},
    author={Padlewski, Piotr and Bain, Max and Henderson, Matthew and Zhu, Zhongkai
    and Relan, Nishant and Pham, Hai and Ong, Donovan and Aleksiev, Kaloyan and Ormazabal, Aitor
    and Phua, Samuel and others},
    journal={arXiv preprint arXiv:2405.02287},
    year={2024}
    }

    Paper: https://arxiv.org/abs/2405.02287
    """

    VIBE_EVAL_HUGGINGFACE_DATASET_NAME: str = "RekaAI/VibeEval"

    SUBJECTS: List[str] = [
        "difficulty-hard",
        "difficulty-normal",
    ]

    name = "vibe_eval"
    description = (
        "Evaluate multimodal models on day-to-day tasks "
        "([Padlewski et al., 2024](https://arxiv.org/abs/2405.02287))."
    )
    tags = ["vision-language", "knowledge", "reasoning"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        instances: List[Instance] = []
        # Process the test set
        for row in tqdm(
            load_dataset(
                self.VIBE_EVAL_HUGGINGFACE_DATASET_NAME,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            if row["category"] != self._subject:
                continue
            example_id: str = row["example_id"].replace("/", "-")
            # Save the image locally
            local_image_path: str = os.path.join(images_path, f"{example_id}.png")
            if not os.path.exists(local_image_path):
                row["image"].convert("RGB").save(local_image_path, "PNG", optimize=True)

            content: List[MediaObject] = [
                MediaObject(location=local_image_path, content_type="image/png"),
                MediaObject(text=row["prompt"], content_type="text/plain"),
            ]
            answer: str = row["reference"]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
