from helm.benchmark.scenarios.scenario import VALID_SPLIT
from .utils_latex import latex_to_image
from helm.benchmark.scenarios.vision_language.image2structure.image2structure_scenario import Image2StructureScenario
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])


class LatexScenario(Image2StructureScenario):
    BASE_PROMPT = "Please provide the LaTeX code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."  # noqa: E501
    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-latex"
    SUBSETS = ["equation", "table", "plot", "algorithm"]

    name = "image2latex"
    description = "Evaluate multimodal models on Latex generation to recreate a provided image"

    def __init__(self, subset: str, recompile_prompt: bool = True, split: str = VALID_SPLIT):
        super().__init__(subset, recompile_prompt, split)

    def compile(self, structure: str, assets_path: str) -> Image.Image:
        image, _ = latex_to_image(structure, assets_path=assets_path, crop=True)
        return image
