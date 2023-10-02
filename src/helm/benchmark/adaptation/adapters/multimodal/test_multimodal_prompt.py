from typing import List
import unittest

from helm.common.multimodal_content import MultimodalContent
from helm.common.file import File, IMAGE_MEDIA_TYPE
from .multimodal_prompt import MultimodalPrompt


class TestMultimodalContent(unittest.TestCase):
    def test_content(self):
        train_instance_blocks: List[MultimodalContent] = [
            MultimodalContent([File("http://dog.png", IMAGE_MEDIA_TYPE, "png"), "Which animal is this?", "dog"]),
            MultimodalContent([File("http://mouse.png", IMAGE_MEDIA_TYPE, "png"), "Which animal is this?", "mouse"]),
        ]
        eval_instance_block = MultimodalContent(
            [File("http://cat.png", IMAGE_MEDIA_TYPE, "png"), "Which animal is this?"]
        )

        prompt = MultimodalPrompt(
            global_prefix="[START]",
            instance_prefix="\n",
            instructions_block="Please answer the following questions about the images.",
            train_instance_blocks=train_instance_blocks,
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.content.content,
            [
                "[START]Please answer the following questions about the images.",
                "\n",
                File("http://dog.png", IMAGE_MEDIA_TYPE, "png"),
                "Which animal is this?",
                "dog",
                "\n",
                File("http://mouse.png", IMAGE_MEDIA_TYPE, "png"),
                "Which animal is this?",
                "mouse",
                "\n",
                File("http://cat.png", IMAGE_MEDIA_TYPE, "png"),
                "Which animal is this?",
            ],
        )

    def test_content_zero_shot(self):
        eval_instance_block = MultimodalContent(
            [File("http://cat.png", IMAGE_MEDIA_TYPE, "png"), "Which animal is this?"]
        )

        prompt = MultimodalPrompt(
            global_prefix="",
            instance_prefix="\n",
            instructions_block="",
            train_instance_blocks=[],
            eval_instance_block=eval_instance_block,
        )
        self.assertEqual(
            prompt.content.content, [File("http://cat.png", IMAGE_MEDIA_TYPE, "png"), "Which animal is this?"]
        )
