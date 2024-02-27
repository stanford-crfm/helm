from typing import List, Tuple

from helm.benchmark.metrics.vision_language.image_metrics import GenerateImageFromCompletionMetric, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.vision_language.image2structure.utils_latex import latex_to_image
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from PIL.Image import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


class LatexMetric(GenerateImageFromCompletionMetric):
    DELIMITERS: List[Tuple[str, str]] = [
        ("```latex", "```"),
        ("```", "```"),
    ]

    def __init__(self, metric_names: List[str], normalize_by_white_score: bool = False):
        super().__init__("latex", metric_names, normalize_by_white_score)

    def compile_completion_into_image(
        self, request_state: RequestState, completion: str, ref_image: Image, eval_cache_path: str
    ) -> Image:
        """Given a completion, parse the LaTeX and compile it into an image."""
        # Get the assets path
        assets_path: str = ""

        # Check for code block delimiters
        # After this completion should be a valid latex code block
        for start, end in self.DELIMITERS:
            if start in completion and end in completion[completion.index(start) + len(start) :]:
                start_index = completion.index(start) + len(start)
                end_index = completion.index(end, start_index)
                completion = completion[start_index:end_index]
                break

        # Convert the latex code to an image
        try:
            image: Image = latex_to_image(completion, assets_path, crop=True)[0]
        except RuntimeError as e:
            # We do not want to catch OptionalDependencyNotInstalled (error with latex installation)
            # because it is a fatal error and should be handled by the user
            raise CompilationError from e

        return image
