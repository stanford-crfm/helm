from typing import List
import unittest

from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.adaptation.adapters.multimodal.multimodal_prompt import MultimodalPrompt


class TestMultimodalContent(unittest.TestCase):
    def test_content(self):
        train_instance_blocks: List[MultimediaObject] = [
            MultimediaObject(
                [
                    MediaObject(location="http://dog.png", content_type="image/png"),
                    MediaObject(text="Which animal is this?", content_type="text/plain"),
                    MediaObject(text="dog", content_type="text/plain"),
                ]
            ),
            MultimediaObject(
                [
                    MediaObject(location="http://mouse.png", content_type="image/png"),
                    MediaObject(text="Which animal is this?", content_type="text/plain"),
                    MediaObject(text="mouse", content_type="text/plain"),
                ]
            ),
        ]
        eval_instance_block = MultimediaObject(
            [
                MediaObject(location="http://cat.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
            ]
        )

        prompt = MultimodalPrompt(
            global_prefix="[START]",
            global_suffix="",
            instance_prefix="\n",
            instructions="Please answer the following questions about the images.",
            train_instance_blocks=train_instance_blocks,
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.multimedia_object.media_objects,
            [
                MediaObject(
                    text="[START]Please answer the following questions about the images.", content_type="text/plain"
                ),
                MediaObject(text="\n", content_type="text/plain"),
                MediaObject(location="http://dog.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
                MediaObject(text="dog", content_type="text/plain"),
                MediaObject(text="\n", content_type="text/plain"),
                MediaObject(location="http://mouse.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
                MediaObject(text="mouse", content_type="text/plain"),
                MediaObject(text="\n", content_type="text/plain"),
                MediaObject(location="http://cat.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
            ],
        )

    def test_content_zero_shot(self):
        eval_instance_block: MultimediaObject = MultimediaObject(
            [
                MediaObject(location="http://cat.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
            ]
        )

        prompt = MultimodalPrompt(
            global_prefix="",
            global_suffix="",
            instance_prefix="\n",
            instructions="",
            train_instance_blocks=[],
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.multimedia_object.media_objects,
            [
                MediaObject(location="http://cat.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
            ],
        )
