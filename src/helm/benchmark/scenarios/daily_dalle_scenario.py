from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT


class DailyDallEScenario(Scenario):
    """
    Prompts from Chad Nelson's Instagram: https://www.instagram.com/dailydall.e
    Chad Nelson was a featured artist on OpenAI's blogpost:
    https://openai.com/blog/dall-e-2-extending-creativity.
    """

    PROMPTS: List[str] = ["TODO"]

    name = "daily_dalle"
    description = (
        "Prompts from [Chad Nelson's Instagram](https://www.instagram.com/dailydall.e/)"
        "Chad Nelson was a featured artist on [OpenAI's blogpost]"
        "(https://openai.com/blog/dall-e-2-extending-creativity)."
    )
    tags = ["text-to-image", "originality"]

    def get_instances(self) -> List[Instance]:
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in self.PROMPTS]
