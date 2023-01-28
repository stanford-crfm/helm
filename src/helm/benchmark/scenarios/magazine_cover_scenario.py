from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT


class MagazineCoverScenario(Scenario):
    """
    Prompts to generate magazine cover photos. Each prompt contains
    a real headline from one of following magazines:

    - TIME
    """

    PROMPTS: List[str] = ["TODO"]

    name = "magazine_cover"
    description = "Prompts to generate magazine cover photos"
    tags = ["text-to-image", "originality"]

    def get_instances(self) -> List[Instance]:
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in self.PROMPTS]
