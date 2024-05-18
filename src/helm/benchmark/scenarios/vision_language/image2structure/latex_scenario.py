from helm.benchmark.scenarios.scenario import VALID_SPLIT
from helm.benchmark.scenarios.vision_language.image2structure.utils_latex import (
    latex_to_image,
    strip_unnecessary_latex_parts,
)
from helm.benchmark.scenarios.vision_language.image2structure.image2structure_scenario import Image2StructureScenario


class LatexScenario(Image2StructureScenario):
    BASE_PROMPT = "Please provide the LaTeX code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."  # noqa: E501
    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-latex"
    SUBSETS = ["equation", "table", "plot", "algorithm", "real"]

    name = "image2latex"
    description = "Evaluate multimodal models on Latex generation to recreate a provided image"

    def __init__(self, subset: str, recompile_prompt: bool = True, split: str = VALID_SPLIT):
        super().__init__(subset, recompile_prompt, split)

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> str:
        image, infos = latex_to_image(structure, assets_path=assets_path, crop=True)
        image.save(destination_path)
        assert "latex_code" in infos
        text: str = strip_unnecessary_latex_parts(infos["latex_code"])
        return text
