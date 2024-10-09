from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class RadiologyScenario(Scenario):
    """
    From "What Does DALL-E 2 Know About Radiology?", DALL-E 2 fails to generate realistic
    X-ray, CT, MRI and ultrasound images. For example, generated images of X-rays have
    smaller bones missing, extra joints added, etc.

    Prompts are in the following format:

    “An x-ray of {head, chest, shoulder, abdomen, pelvis, hand, knee, ankle}” => 8 prompts
    “An MRI of the {heart, liver, kidney}” => 3 prompts
    “A CT of the {heart, liver, kidney}” => 3 prompts
    “Ultrasound of the {heart, liver, kidney}” => 3 prompts

    Paper: https://arxiv.org/abs/2209.13696
    """

    X_RAY_ANATOMICAL_AREA: List[str] = ["head", "chest", "shoulder", "abdomen", "pelvis", "hand", "knee", "ankle"]
    ORGANS: List[str] = ["heart", "liver", "kidney"]

    name = "radiology"
    description = "Prompts to generate radiological images ([paper](https://arxiv.org/abs/2209.13696))."
    tags = ["text-to-image", "knowledge"]

    def get_instances(self, _) -> List[Instance]:
        prompts: List[str] = []

        for anatomical_area in self.X_RAY_ANATOMICAL_AREA:
            prompts.append(f"An x-ray of {anatomical_area}")

        for organ in self.ORGANS:
            prompts.append(f"An MRI of the {organ}")
            prompts.append(f"A CT of the {organ}")
            prompts.append(f"Ultrasound of the {organ}")

        assert len(prompts) == 17, "Invalid number of prompts"
        # There are no reference images
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in prompts]
