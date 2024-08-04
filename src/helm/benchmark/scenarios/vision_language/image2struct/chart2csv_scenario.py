from typing import List
import os

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject


class Chart2CSVScenario(Scenario):
    """
    Chart to CSV
    """

    name = "chart2csv"
    description = "Convert a chart to CSV."
    tags = ["vision-language", "image2struct"]

    def get_instances(self, output_path: str) -> List[Instance]:
        assert os.path.exists(output_path), f"Dataset does not exist at {output_path}"
        instances: List[Instance] = []

        charts_path: str = os.path.join(output_path, "charts")
        ground_truths_path: str = os.path.join(output_path, "groundtruth")

        for chart_file in os.listdir(charts_path):
            if not chart_file.endswith(".png"):
                continue

            chart_path: str = os.path.join(charts_path, chart_file)
            ground_truth_file: str = chart_file.replace(".png", ".psv")
            ground_truth_path: str = os.path.join(ground_truths_path, ground_truth_file)
            assert os.path.exists(ground_truth_path), f"Ground truth does not exist at {ground_truth_path}"

            content: List[MediaObject] = [
                MediaObject(location=chart_path, content_type="image/png"),
            ]
            with open(ground_truth_path, "r") as file:
                ground_truth: str = file.read().replace("|", ",")

            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(text=ground_truth), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
