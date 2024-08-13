import os
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    ALL_SPLITS,
    VALID_SPLIT,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class AOKVQAScenario(Scenario):
    """
    A crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense
    and world knowledge to answer.

    @misc{schwenk2022aokvqa,
          title={A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge},
          author={Dustin Schwenk and Apoorv Khandelwal and Christopher Clark and Kenneth Marino and Roozbeh Mottaghi},
          year={2022},
          eprint={2206.01718},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

    Paper: https://arxiv.org/abs/2206.01718
    Website: https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
    """

    HF_DATASET_NAME: str = "HuggingFaceM4/A-OKVQA"

    name = "a_okvqa"
    description = (
        "A crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of "
        "commonsense and world knowledge to answer ([Schwenk et al., 2022](https://arxiv.org/abs/2206.01718))."
    )
    tags = ["vision-language", "knowledge", "reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        instances: List[Instance] = []
        for helm_split in ALL_SPLITS:
            if helm_split == TEST_SPLIT:
                # The examples in the test split does not have answers
                continue

            split = "validation" if helm_split == VALID_SPLIT else helm_split

            for row in tqdm(load_dataset(self.HF_DATASET_NAME, cache_dir=output_path, split=split)):
                image_filename: str = f"{row['question_id']}.jpg"
                local_image_path: str = os.path.join(images_path, image_filename)
                image = row["image"]
                if not os.path.exists(local_image_path):
                    image.save(local_image_path)

                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/jpeg"),
                    MediaObject(text=row["question"], content_type="text/plain"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[
                            Reference(Output(text=choice), tags=[CORRECT_TAG] if i == row["correct_choice_idx"] else [])
                            for i, choice in enumerate(row["choices"])
                        ],
                        split=helm_split,
                    )
                )

        return instances
