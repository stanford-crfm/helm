from typing import List, Tuple, Dict, Any

from helm.benchmark.annotation.image2struct.image_compiler_annotator import ImageCompilerAnnotator, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.vision_language.image2struct.utils_latex import (
    latex_to_image,
    strip_unnecessary_latex_parts,
)

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


class LatexCompilerAnnotator(ImageCompilerAnnotator):
    """Annotator that compiles the text completions into a LaTeX document."""

    name: str = "latex_compiler"

    # Delimiters for the code block
    DELIMITERS: List[Tuple[str, str]] = [
        ("```latex", "```"),
        ("```", "```"),
    ]

    def postprocess_infos(self, infos: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess the infos."""
        annotations = super().postprocess_infos(infos)
        assert "latex_code" in annotations, "The latex_code field should be present in the infos"
        annotations["text"] = strip_unnecessary_latex_parts(annotations["latex_code"])
        return annotations

    def compile_completion_into_image(
        self, request_state: RequestState, completion_text: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Given a completion, parse the LaTeX and compile it into an image."""
        # Get the assets path
        assets_path: str = ""

        # Check for code block delimiters
        # After this completion should be a valid latex code block
        for start, end in self.DELIMITERS:
            if start in completion_text and end in completion_text[completion_text.index(start) + len(start) :]:
                start_index = completion_text.index(start) + len(start)
                end_index = completion_text.index(end, start_index)
                completion_text = completion_text[start_index:end_index]
                break

        # Convert the latex code to an image
        try:
            image, infos = latex_to_image(completion_text, assets_path, crop=True)
        except RuntimeError as e:
            # We do not want to catch OptionalDependencyNotInstalled (error with latex installation)
            # because it is a fatal error and should be handled by the user
            raise CompilationError(str(e)) from e

        return image, infos
