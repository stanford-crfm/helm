from helm.benchmark.scenarios.scenario import VALID_SPLIT
from helm.benchmark.scenarios.vision_language.image2structure.image2structure_scenario import Image2StructureScenario


class MusicSheetScenario(Image2StructureScenario):
    BASE_PROMPT = (
        "Please generate the Lilypond code to generate a music sheet that looks like this image as much as feasibly possible.\n"  # noqa: E501
        "This music sheet was created by me, and I would like to recreate it using Lilypond."
    )
    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-musicsheet"
    SUBSETS = ["music"]

    name = "image2musicsheet"
    description = "Evaluate multimodal models on Lilypond generation to recreate a provided image"

    def __init__(self, subset: str, recompile_prompt: bool = True, split: str = VALID_SPLIT):
        super().__init__(subset, recompile_prompt, split)

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> str:
        raise Exception("Music sheets have no ground truth, compilation is not possible")
