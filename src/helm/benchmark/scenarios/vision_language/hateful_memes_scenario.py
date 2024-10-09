import os.path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class HatefulMemesScenario(Scenario):
    """
    The Hateful Memes Challenge and Dataset is a competition and open source dataset designed to
    measure progress in multimodal vision-and-language classification.

    In order for AI to become a more effective tool for detecting hate speech, it must be able to understand
    content the way people do: holistically. When viewing a meme, for example, we don’t think about the words
    and photo independently of each other; we understand the combined meaning. This is extremely challenging
    for machines, however, because it means they can’t analyze the text and the image separately. They must
    combine these different modalities and understand how the meaning changes when they are presented together.

    @article{kiela2020hateful,
      title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
      author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh,
      Amanpreet and Ringshia, Pratik and  Testuggine, Davide},
      journal={arXiv preprint arXiv:2005.04790},
      year={2020}
    }

    Paper: https://arxiv.org/abs/2005.04790
    Website: https://ai.meta.com/tools/hatefulmemes/
    """

    IMAGE_URL: str = (
        "https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/main/{image_path}?download=true"
    )

    # This question is from the example on the website
    QUESTION: str = "Is this meme mean?"

    name = "hateful_memes"
    description = "Identify if a meme is hateful or not ([Kiela et al., 2020](https://arxiv.org/abs/2005.04790))."
    tags = ["vision-language"]

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(os.path.join(output_path, "img"))

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            for row in tqdm(
                load_dataset(
                    "neuralcatcher/hateful_memes",
                    split="validation" if split == VALID_SPLIT else split,
                    cache_dir=output_path,
                )
            ):
                # Download the meme
                image_path: str = row["img"]
                local_image_path: str = os.path.join(output_path, image_path)
                ensure_file_downloaded(
                    source_url=self.IMAGE_URL.format(image_path=image_path),
                    target_path=local_image_path,
                    unpack=False,
                )
                # Some examples are missing images. Skip those for now
                if not os.path.exists(local_image_path) or os.path.getsize(local_image_path) == 0:
                    continue

                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/jpeg"),
                    MediaObject(text=self.QUESTION, content_type="text/plain"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[
                            Reference(Output(text="Yes"), tags=[CORRECT_TAG] if row["label"] == 1 else []),
                            Reference(Output(text="No"), tags=[CORRECT_TAG] if row["label"] == 0 else []),
                        ],
                        split=split,
                    )
                )

        return instances
