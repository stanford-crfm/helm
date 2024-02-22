from typing import List, Tuple
import json
import os

from helm.benchmark.metrics.vision_language.image_metrics import GenerateImageFromCompletionMetric, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.vision_language.image2structure.webpage.driver import (
    ScreenshotOptions,
)
from helm.benchmark.scenarios.vision_language.image2structure.webpage_scenario import (
    serve_and_take_screenshot,
    list_assets,
    copy_assets,
)

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


class WebpageMetric(GenerateImageFromCompletionMetric):
    DELIMITERS: List[Tuple[str, str]] = [
        ("[", "]"),
    ]

    def __init__(
        self,
        metric_names: List[str],
        normalize_by_white_score: bool = False,
        screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    ):
        super().__init__("webpage", metric_names, normalize_by_white_score)
        self._screenshot_options = screenshot_options

    def compile_completion_into_image(
        self, request_state: RequestState, completion: str, ref_image: Image.Image
    ) -> Image.Image:
        """Given a completion, parse the code and compile it into an image."""
        repo_path: str = ""

        # Check for code block delimiters
        # After this completion should be a valid json object
        for start, end in self.DELIMITERS:
            if start in completion and end in completion[completion.index(start) + len(start) :]:
                start_index = completion.index(start) + len(start)
                end_index = completion.index(end, start_index)
                completion = completion[start_index:end_index]
                break

        # Parse code into json object
        structure: dict
        try:
            structure = json.loads(completion)
        except json.JSONDecodeError as e:
            raise CompilationError from e

        # Create each file in a temporary directory
        for filename, content in structure.items():
            # Create parent directories if they do not exist
            parent_dir = os.path.join(repo_path, os.path.dirname(filename))
            os.makedirs(parent_dir, exist_ok=True)
            with open(os.path.join(repo_path, filename), "w") as f:
                f.write(content)

        # Copy the assets
        references = request_state.instance.references
        assert len(references) > 0, "No references found"
        original_repo_path = references[0].output.text
        print(f"original_repo_path: {original_repo_path}")
        asset_paths: List[str] = list_assets(original_repo_path, ["png", "jpg", "jpeg", "gif", "svg", "webp", "ico"])
        copy_assets(original_repo_path, repo_path, asset_paths)

        # Save the screenshot, loads the image and remove the file
        destination_path: str = os.path.join(repo_path, "output.png")
        serve_and_take_screenshot(repo_path, destination_path, self._screenshot_options)
        image: Image.Image = Image.open(destination_path)
        os.remove(destination_path)
        return image
