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


class SheetMusic2LilyPondScenario(Scenario):
    """
    Sheet music to LilyPond scenario

    LilyPond is a powerful music engraving program that produces high-quality sheet music. It allows
    musicians to create elegant and readable scores, following the best traditions of
    classical music engraving.
    """

    name = "sheetmusic2lilypond"
    description = "Convert sheet music to LilyPond"
    tags = ["vision-language", "sheetmusic2lilypond"]

    def get_instances(self, output_path: str) -> List[Instance]:
        assert os.path.exists(output_path), f"Dataset does not exist at {output_path}"
        instances: List[Instance] = []

        for image_file in os.listdir(output_path):
            if not image_file.endswith(".png"):
                continue

            image_path: str = os.path.join(output_path, image_file)
            content: List[MediaObject] = [
                MediaObject(location=image_path, content_type="image/png"),
            ]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(multimedia_content=MultimediaObject(content)), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
