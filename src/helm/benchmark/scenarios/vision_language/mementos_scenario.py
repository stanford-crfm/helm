import os.path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

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
from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.common.media_object import MediaObject, MultimediaObject


class MementosScenario(Scenario):
    """
    Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences

    We introduces Mementos, a new benchmark designed to assess MLLMs' sequential image reasoning abilities. Mementos
    features 4,761 diverse image sequences with varying lengths.

    @misc{wang2024mementos,
    title={Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences},
    author={Xiyao Wang and Yuhang Zhou and Xiaoyu Liu and Hongjin Lu and Yuancheng Xu and Feihong He and Jaehong Yoon
    and Taixi Lu and Gedas Bertasius and Mohit Bansal and Huaxiu Yao and Furong Huang},
    year={2024},
    eprint={2401.10529},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

    Paper: https://arxiv.org/abs/2401.10529
    """

    MEMENTOS_HUGGINGFACE_DATASET_NAME: str = "RussWang96/unofficial_mementos_dataset"

    IMAGE_URL: str = (
        "https://huggingface.co/datasets/RussWang96/unofficial_mementos_dataset/resolve/main/"
        + "{subject}/{split}/{file_name}?download=true"
    )

    DATA_FILES: str = "{subject}/{split}/metadata.csv"

    QUESTION_PROMPT: str = (
        "Write a description for the given image sequence in a single paragraph, what is happening in this episode?"
    )

    SUBJECTS: List[str] = ["comics", "dailylife", "robotics"]

    name = "mementos"
    description = (
        "A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences"
        " ([Wang et al., 2024](https://arxiv.org/abs/2401.10529))."
    )
    tags = ["vision-language"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        subject_output_path: str = os.path.join(output_path, self._subject)
        ensure_directory_exists(subject_output_path)

        data_files = {
            split: self.DATA_FILES.format(subject=self._subject, split=split) for split in [TRAIN_SPLIT, TEST_SPLIT]
        }
        instances: List[Instance] = []

        for split in [TRAIN_SPLIT, TEST_SPLIT]:
            cur_output_path = os.path.join(subject_output_path, split)
            ensure_directory_exists(cur_output_path)

            # Process the test set
            for row in tqdm(
                load_dataset(
                    self.MEMENTOS_HUGGINGFACE_DATASET_NAME.format(subject=self._subject),
                    data_files=data_files,
                    split=split,
                    cache_dir=cur_output_path,
                )
            ):
                # Download the image
                file_name: str = row["file_name"]
                local_image_path: str = os.path.join(cur_output_path, file_name)
                ensure_file_downloaded(
                    source_url=self.IMAGE_URL.format(subject=self._subject, split=split, file_name=file_name),
                    target_path=local_image_path,
                    unpack=False,
                )

                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/png"),
                    MediaObject(text=self.QUESTION_PROMPT, content_type="text/plain"),
                ]
                answer: str = row["description"]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

            print()

        return instances


def main():
    scenario = MementosScenario("robotics")
    instances = scenario.get_instances("output")
    print(instances)


if __name__ == "__main__":
    main()
