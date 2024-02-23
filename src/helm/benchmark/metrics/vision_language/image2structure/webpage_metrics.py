from typing import List, Tuple, Optional
import json
import os
import shutil

from helm.benchmark.metrics.vision_language.image_metrics import GenerateImageFromCompletionMetric, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.vision_language.image2structure.webpage.driver import (
    ScreenshotOptions,
)
from helm.benchmark.scenarios.vision_language.image2structure.webpage_scenario import serve_and_take_screenshot
from helm.benchmark.scenarios.scenario import ASSET_NAME_TAG, ASSET_PATH_TAG

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["images"])


class WebpageMetric(GenerateImageFromCompletionMetric):
    DELIMITERS: List[Tuple[str, str]] = []

    def __init__(
        self,
        metric_names: List[str],
        normalize_by_white_score: bool = False,
        screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    ):
        super().__init__("webpage", metric_names, normalize_by_white_score)
        self._screenshot_options = screenshot_options

    def compile_completion_into_image(
        self, request_state: RequestState, completion: str, ref_image: Image.Image, eval_cache_path: str
    ) -> Image.Image:
        """Given a completion, parse the code and compile it into an image."""
        repo_path: str = os.path.join(eval_cache_path, "repo")

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
            raise CompilationError(f"Failed to parse the completion as a JSON object: {e}") from e

        # Copy the assets
        assets_paths: List[str] = []
        assets_names: List[str] = []
        for reference in request_state.instance.references:
            if ASSET_PATH_TAG in reference.tags:
                assert reference.output.multimedia_content is not None
                for media_object in reference.output.multimedia_content.media_objects:
                    assert media_object.is_local_file
                    assert media_object.is_type("image")
                    assert type(media_object.location) == str
                    assets_paths.append(media_object.location)
            if ASSET_NAME_TAG in reference.tags:
                assert reference.output.multimedia_content is not None
                for media_object in reference.output.multimedia_content.media_objects:
                    assert media_object.is_type("text")
                    assert type(media_object.text) == str
                    assets_names.append(media_object.text)
        assert len(assets_paths) == len(assets_names)
        for asset_path, asset_name in zip(assets_paths, assets_names):
            dest_path: str = os.path.join(repo_path, asset_name)
            # Make sure the parent directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(asset_path, dest_path)
            # os.symlink(asset_path, dest_path)

        # Create each file in a temporary directory
        if not isinstance(structure, list):
            raise CompilationError("The completion should be a list of files")
        for item in structure:
            filename: Optional[str] = item.get("filename")
            content: Optional[str] = item.get("content")
            if filename is None or content is None:
                raise CompilationError("Each file should have a valid filename and content")
            # Create parent directories if they do not exist
            if filename in assets_names:
                # Some models will include assets in their response like this:
                # {
                #     "filename": "chmber.jpg",
                #     "content": "The content of the chmber.jpg file is a binary image and cannot be displayed as text."
                # }
                # In this case, we skip the file creation
                continue
            parent_dir = os.path.join(repo_path, os.path.dirname(filename))
            os.makedirs(parent_dir, exist_ok=True)
            with open(os.path.join(repo_path, filename), "w") as f:
                f.write(content)

        # Save the screenshot, loads the image and remove the file
        destination_path: str = os.path.join(repo_path, "output.png")
        serve_and_take_screenshot(repo_path, destination_path, self._screenshot_options)
        image: Image.Image = Image.open(destination_path)

        # Delete the repository
        shutil.rmtree(repo_path)

        return image
