from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT


class LogosScenario(Scenario):
    """
    Prompts to generate company logos.
    """

    PROMPTS: List[str] = ["TODO"]

    name = "logos"
    description = "Prompts to generate brand/company logos"
    tags = ["text-to-image", "originality"]

    def get_instances(self) -> List[Instance]:
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in self.PROMPTS]
