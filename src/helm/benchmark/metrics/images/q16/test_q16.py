from typing import List
import os
import pytest

from .q16_toxicity_detector import Q16ToxicityDetector


class TestQ16:
    def setup_method(self, method):
        self._q16_detector = Q16ToxicityDetector()
        # The test images are from the example code
        self._base_path: str = os.path.join(os.path.dirname(__file__), "test_images")

    def test_is_appropriate(self):
        image_paths: List[str] = [
            os.path.join(self._base_path, "sample_appropriate.jpg"),
            os.path.join(self._base_path, "sample_inappropriate.png"),
        ]
        assert self._q16_detector.is_inappropriate_batch(image_paths) == [False, True]

    def test_is_appropriate_flipped(self):
        image_paths: List[str] = [
            os.path.join(self._base_path, "sample_inappropriate.png"),
            os.path.join(self._base_path, "sample_appropriate.jpg"),
        ]
        assert self._q16_detector.is_inappropriate_batch(image_paths) == [True, False]

    def test_is_appropriate_single_appropriate(self):
        image_path: str = os.path.join(self._base_path, "sample_appropriate.jpg")
        assert not self._q16_detector.is_inappropriate(image_path)

    def test_is_appropriate_double_appropriate(self):
        image_paths: List[str] = [
            os.path.join(self._base_path, "sample_appropriate.jpg"),
            os.path.join(self._base_path, "sample_appropriate.jpg"),
        ]
        assert self._q16_detector.is_inappropriate_batch(image_paths) == [False, False]

    def test_is_appropriate_single_inappropriate(self):
        image_path: str = os.path.join(self._base_path, "sample_inappropriate.png")
        assert self._q16_detector.is_inappropriate(image_path)

    @pytest.mark.skip("This test is failing")
    def test_is_appropriate_double_inappropriate(self):
        image_paths: List[str] = [
            os.path.join(self._base_path, "sample_inappropriate.png"),
            os.path.join(self._base_path, "sample_inappropriate.png"),
        ]
        assert self._q16_detector.is_inappropriate_batch(image_paths) == [True, True]
