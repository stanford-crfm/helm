from typing import List, Tuple, Optional, Dict, Any
import json
import os
import shutil
import threading

from helm.benchmark.annotation.image2struct.image_compiler_annotator import ImageCompilerAnnotator, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.vision_language.image2struct.webpage.driver import ScreenshotOptions
from helm.benchmark.scenarios.vision_language.image2struct.webpage.utils import convert_html_to_text
from helm.benchmark.scenarios.vision_language.image2struct.webpage_scenario import serve_and_take_screenshot
from helm.benchmark.scenarios.scenario import ASSET_NAME_TAG, ASSET_PATH_TAG
from helm.common.general import ensure_directory_exists
from helm.common.cache import CacheConfig

try:
    from PIL import Image
    from html2text import HTML2Text
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


class WebpageCompilerAnnotator(ImageCompilerAnnotator):
    """Annotator that compiles the text completions into a webpage
    And takes a screenshot of the webpage."""

    name: str = "webpage_compiler"

    # Delimiters for the code block
    DELIMITERS: List[Tuple[str, str]] = [
        ("```json", "```"),
        ("```", "```"),
    ]

    def __init__(self, cache_config: CacheConfig, file_storage_path: str):
        super().__init__(cache_config, file_storage_path)
        self._html2text = HTML2Text()
        self._html2text.ignore_links = True

    def postprocess_infos(self, infos: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess the infos."""
        annotations = super().postprocess_infos(infos)
        assert "html" in annotations, "The html field should be present in the infos"
        annotations["text"] = convert_html_to_text(self._html2text, infos["html"])
        return annotations

    def compile_completion_into_image(
        self, request_state: RequestState, completion_text: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Given a completion, parse the code and compile it into an image and return the image and the infos."""
        # Create a temporary directory to store the files
        cache_config: CacheConfig = self._cache.config
        repo_path: str = "prod_env/tmp"
        if hasattr(cache_config, "path"):
            repo_path = os.path.join(os.path.dirname(cache_config.path), "tmp")
        # Make the repo path thread safe by adding the thread id
        repo_path = f"{repo_path}_{threading.get_ident()}"
        ensure_directory_exists(repo_path)

        # Check for code block delimiters
        # After this completion should be a valid json object
        for start, end in self.DELIMITERS:
            if start in completion_text and end in completion_text[completion_text.index(start) + len(start) :]:
                start_index = completion_text.index(start) + len(start)
                end_index = completion_text.index(end, start_index)
                completion_text = completion_text[start_index:end_index]
                break

        # Parse code into json object
        structure: dict
        try:
            structure = json.loads(completion_text)
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
        infos: Dict[str, Any] = serve_and_take_screenshot(repo_path, destination_path, ScreenshotOptions())
        image: Image.Image = Image.open(destination_path)

        # Delete the repository
        shutil.rmtree(repo_path, ignore_errors=True)

        return image, infos
