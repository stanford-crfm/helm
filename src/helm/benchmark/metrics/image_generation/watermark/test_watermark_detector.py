from typing import List
import os

from helm.benchmark.metrics.image_generation.watermark.watermark_detector import WatermarkDetector


def test_compute_watermark_probability():
    watermark_detector = WatermarkDetector()

    # These test images are from https://github.com/LAION-AI/LAION-5B-WatermarkDetection
    base_path: str = os.path.join(os.path.dirname(__file__), "test_images")
    clear_image_path: str = os.path.join(base_path, "clear_example.png")
    watermark_image_path: str = os.path.join(base_path, "watermark_example.png")

    has_watermarks: List[bool] = watermark_detector.has_watermark([clear_image_path, watermark_image_path])[0]
    assert has_watermarks == [False, True]
