import unittest

from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.key_value_store import request_to_key
from helm.common.mongo_key_value_store import MongoKeyValueStore


class TestMediaObject(unittest.TestCase):
    def test_add_textual_prefix(self):
        multimodal_object = MultimediaObject([MediaObject(text="b", content_type="text/plain")])
        multimodal_object = multimodal_object.add_textual_prefix("a")
        self.assertEqual(multimodal_object.text, "ab")

    def test_add_textual_prefix_with_file(self):
        multimodal_object = MultimediaObject([MediaObject(location="http://some_image.png", content_type="image/png")])
        multimodal_object = multimodal_object.add_textual_prefix("a")
        self.assertEqual(
            multimodal_object.media_objects,
            [
                MediaObject(text="a", content_type="text/plain"),
                MediaObject(location="http://some_image.png", content_type="image/png"),
            ],
        )

    def test_add_textual_suffix(self):
        multimodal_object = MultimediaObject([MediaObject(text="a", content_type="text/plain")])
        multimodal_object = multimodal_object.add_textual_suffix("b")
        self.assertEqual(multimodal_object.text, "ab")

    def test_add_textual_suffix_with_file(self):
        multimodal_object = MultimediaObject([MediaObject(location="http://some_image.png", content_type="image/png")])
        multimodal_object = multimodal_object.add_textual_suffix("b")
        self.assertEqual(
            multimodal_object.media_objects,
            [
                MediaObject(location="http://some_image.png", content_type="image/png"),
                MediaObject(text="b", content_type="text/plain"),
            ],
        )

    def test_canonicalize_key(self):
        obj = MediaObject(text="Text", content_type="text/plain")
        MongoKeyValueStore._canonicalize_key({"media_object": obj})

    def test_request_to_key(self):
        obj = MediaObject(text="Text", content_type="text/plain")
        request_to_key({"media_object": obj})
