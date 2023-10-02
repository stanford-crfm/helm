import unittest

from .file import File, IMAGE_MEDIA_TYPE
from .multimodal_content import MultimodalContent


class TestMultimodalContent(unittest.TestCase):
    def test_add_textual_prefix(self):
        multimodal_content = MultimodalContent(["b"])
        multimodal_content = multimodal_content.add_textual_prefix("a")
        self.assertEqual(multimodal_content.content, ["ab"])

    def test_add_textual_prefix_with_file(self):
        multimodal_content = MultimodalContent([File("http://some_image.png", IMAGE_MEDIA_TYPE, "png")])
        multimodal_content = multimodal_content.add_textual_prefix("a")
        self.assertEqual(multimodal_content.content, ["a", File("http://some_image.png", IMAGE_MEDIA_TYPE, "png")])

    def test_add_textual_suffix(self):
        multimodal_content = MultimodalContent(["a"])
        multimodal_content = multimodal_content.add_textual_suffix("b")
        self.assertEqual(multimodal_content.content, ["ab"])

    def test_add_textual_suffix_with_file(self):
        multimodal_content = MultimodalContent([File("http://some_image.png", IMAGE_MEDIA_TYPE, "png")])
        multimodal_content = multimodal_content.add_textual_suffix("b")
        self.assertEqual(multimodal_content.content, [File("http://some_image.png", IMAGE_MEDIA_TYPE, "png"), "b"])

    def test_combine(self):
        multimodal_content = MultimodalContent(["a"])
        multimodal_content = multimodal_content.combine(MultimodalContent(["b"]))
        self.assertEqual(multimodal_content.content, ["a", "b"])
