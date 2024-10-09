import os
from typing import List

import pandas as pd

from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded, shell
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, CORRECT_TAG, TEST_SPLIT


class CUB200Scenario(Scenario):
    """
    Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset,
    a challenging dataset of 200 bird species.

    Number of categories: 200
    Number of images: 11,788
    Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box

    Paper: https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf
    Website: http://www.vision.caltech.edu/datasets/cub_200_2011

    We use the version from "AttnGAN: Fine-Grained Text to Image Generation with Attentional
    Generative Adversarial Networks" where 10 captions are included for each image.
    The sizes of the splits are as follows:

    Train: 8,855 examples
    Test: 2,933 examples

    Paper: https://arxiv.org/abs/1711.10485
    Website: https://github.com/taoxugit/AttnGAN
    """

    IMAGES_DOWNLOAD_URL: str = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    CAPTIONS_DOWNLOAD_URL: str = "https://drive.google.com/uc?export=download&id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ"

    name = "cub200"
    description = (
        "Caltech-UCSD Birds-200-2011 is a challenging dataset of 200 bird species with 10 captions for each bird"
        "([paper](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf), "
        "[paper](https://arxiv.org/abs/1711.10485))."
    )
    tags = ["text-to-image", "image-to-text"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the images
        images_path: str = os.path.join(output_path, "images")
        ensure_file_downloaded(
            source_url=self.IMAGES_DOWNLOAD_URL,
            target_path=images_path,
            unpack=True,
            unpack_type="untar",
        )
        images_path = os.path.join(images_path, "CUB_200_2011", "images")

        # Download the captions
        captions_path: str = os.path.join(output_path, "captions")
        ensure_file_downloaded(
            source_url=self.CAPTIONS_DOWNLOAD_URL,
            target_path=captions_path,
            unpack=True,
            unpack_type="unzip",
        )
        captions_path = os.path.join(captions_path, "birds")
        text_path: str = os.path.join(captions_path, "text")
        if not os.path.exists(text_path):
            shell(["unzip", os.path.join(captions_path, "text.zip"), "-d", captions_path])

        # Get the text examples. Each example has an image file and text file with 10 captions
        test_filenames_path: str = os.path.join(captions_path, "test", "filenames.pickle")
        test_filenames: List[str] = pd.read_pickle(test_filenames_path)
        assert len(test_filenames) == 2_933, "Expected 2,933 examples in the test split."

        instances: List[Instance] = []
        for file_name in test_filenames:
            image_path: str = os.path.join(images_path, f"{file_name}.jpg")
            assert os.path.exists(image_path), f"Expected an image at path: {image_path}"

            caption_path: str = os.path.join(text_path, f"{file_name}.txt")
            with open(caption_path, "r") as f:
                captions: List[str] = [caption_line.rstrip() for caption_line in f if caption_line.rstrip()]
                assert len(captions) == 10, f"Expected 10 captions at path: {caption_path}"

            for caption in captions:
                content: MultimediaObject = MultimediaObject(
                    [MediaObject(content_type="image/jpeg", location=image_path)]
                )
                instance = Instance(
                    Input(text=caption),
                    references=[Reference(Output(multimedia_content=content), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances
