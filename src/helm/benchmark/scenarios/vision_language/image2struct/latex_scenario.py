from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import ScenarioMetadata
from helm.benchmark.scenarios.vision_language.image2struct.utils_latex import (
    latex_to_image,
    strip_unnecessary_latex_parts,
)
from helm.benchmark.scenarios.vision_language.image2struct.image2struct_scenario import Image2StructureScenario


class LatexScenario(Image2StructureScenario):
    BASE_PROMPT = "Please provide the LaTeX code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."  # noqa: E501
    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-latex"
    SUBSETS = ["equation", "table", "plot", "algorithm", "wild", "wild_legacy"]

    name = "image2latex"
    description = "Evaluate multimodal models on Latex generation to recreate a provided image"

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> str:
        image, infos = latex_to_image(structure, assets_path=assets_path, crop=True)
        image.save(destination_path)
        assert "latex_code" in infos
        text: str = strip_unnecessary_latex_parts(infos["latex_code"])
        return text

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="image2latex",
            display_name="Image2LaTeX",
            description="The Image2LaTeX benchmark for converting images of mathematical equations, "
            "tables. algorithms and tikz to LaTeX.",
            taxonomy=TaxonomyInfo(
                task="image-to-text",
                what="mathematical equations, tables, algorithms, tikz",
                when="2024",
                who="dataset authors",
                language="English",
            ),
            main_metric="earth_mover_similarity",
            main_split="valid",
        )
