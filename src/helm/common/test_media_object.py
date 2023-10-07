import unittest

from .media_object import MediaObject, add_textual_prefix, add_textual_suffix, extract_text


class TestMediaObject(unittest.TestCase):
    def test_add_textual_prefix(self):
        multimodal_content = [MediaObject(text="b", content_type="text/plain")]
        multimodal_content = add_textual_prefix(multimodal_content, "a")
        self.assertEqual(extract_text(multimodal_content), "ab")

    def test_add_textual_prefix_with_file(self):
        multimodal_content = [MediaObject(location="http://some_image.png", content_type="image/png")]
        multimodal_content = add_textual_prefix(multimodal_content, "a")
        self.assertEqual(
            multimodal_content,
            [
                MediaObject(text="a", content_type="text/plain"),
                MediaObject(location="http://some_image.png", content_type="image/png"),
            ],
        )

    def test_add_textual_suffix(self):
        multimodal_content = [MediaObject(text="a", content_type="text/plain")]
        multimodal_content = add_textual_suffix(multimodal_content, "b")
        self.assertEqual(extract_text(multimodal_content), "ab")

    def test_add_textual_suffix_with_file(self):
        multimodal_content = [MediaObject(location="http://some_image.png", content_type="image/png")]
        multimodal_content = add_textual_suffix(multimodal_content, "b")
        self.assertEqual(
            multimodal_content,
            [
                MediaObject(location="http://some_image.png", content_type="image/png"),
                MediaObject(text="b", content_type="text/plain"),
            ],
        )
