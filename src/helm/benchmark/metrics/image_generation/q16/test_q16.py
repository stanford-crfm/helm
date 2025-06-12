import os
import pytest

from helm.benchmark.metrics.image_generation.q16.q16_toxicity_detector import Q16ToxicityDetector


@pytest.mark.skip(reason="Skipping due to flakiness.")
class TestQ16:
    def setup_method(self, method):
        self._q16_detector = Q16ToxicityDetector()
        # The test images are from the example code
        self._base_path: str = os.path.join(os.path.dirname(__file__), "test_images")

    def test_is_appropriate_single_appropriate(self):
        image_path: str = os.path.join(self._base_path, "sample_appropriate.jpg")
        assert not self._q16_detector.is_inappropriate(image_path)

    def test_is_appropriate_single_inappropriate(self):
        image_path: str = os.path.join(self._base_path, "sample_inappropriate.png")
        assert self._q16_detector.is_inappropriate(image_path)
