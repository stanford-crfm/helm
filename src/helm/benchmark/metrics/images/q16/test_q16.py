from typing import List
import os

from .q16_toxicity_detector import Q16ToxicityDetector


def test_is_appropriate():
    # These test images are from the example code
    base_path: str = os.path.join(os.path.dirname(__file__), "test_images")
    image_paths: List[str] = [
        os.path.join(base_path, "sample_appropriate.jpg"),
        os.path.join(base_path, "sample_inappropriate.png"),
    ]

    q16_detector = Q16ToxicityDetector()
    assert q16_detector.is_inappropriate(image_paths) == [False, True]
