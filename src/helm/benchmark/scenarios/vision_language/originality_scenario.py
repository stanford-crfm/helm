import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject


class OriginalityScenario(Scenario):
    """
    Test the originality of VLMs. Work in progress.
    """

    name = "originality_vlm"
    description = "Test the originality of VLMs"
    tags = ["vision-language", "originality"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        content: List[MediaObject] = [
            MediaObject(location=os.path.join(output_path, "story.png"), content_type="image/png"),
            MediaObject(text="Tell a creative story explaining this scene.", content_type="text/plain"),
        ]
        instances.append(
            Instance(
                Input(multimedia_content=MultimediaObject(content)),
                references=[],
                split=TEST_SPLIT,
            )
        )
        return instances
