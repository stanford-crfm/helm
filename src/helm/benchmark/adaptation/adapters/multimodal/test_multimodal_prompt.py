from typing import List
import unittest

from helm.common.media_object import MediaObject
from .multimodal_prompt import MultimodalPrompt


class TestMultimodalContent(unittest.TestCase):
    def test_content(self):
        train_instance_blocks: List[List[MediaObject]] = [
            [
                MediaObject(location="http://dog.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
                MediaObject(text="dog", content_type="text/plain"),
            ],
            [
                MediaObject(location="http://mouse.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
                MediaObject(text="mouse", content_type="text/plain"),
            ],
        ]
        eval_instance_block: List[MediaObject] = [
            MediaObject(location="http://cat.png", content_type="image/png"),
            MediaObject(text="Which animal is this?", content_type="text/plain"),
        ]

        prompt = MultimodalPrompt(
            global_prefix="[START]",
            instance_prefix="\n",
            instructions_block="Please answer the following questions about the images.",
            train_instance_blocks=train_instance_blocks,
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.content,
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
        eval_instance_block: List[MediaObject] = [
            MediaObject(location="http://cat.png", content_type="image/png"),
            MediaObject(text="Which animal is this?", content_type="text/plain"),
        ]

        prompt = MultimodalPrompt(
            global_prefix="",
            instance_prefix="\n",
            instructions_block="",
            train_instance_blocks=[],
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.content,
            [
                MediaObject(location="http://cat.png", content_type="image/png"),
                MediaObject(text="Which animal is this?", content_type="text/plain"),
            ],
        )
