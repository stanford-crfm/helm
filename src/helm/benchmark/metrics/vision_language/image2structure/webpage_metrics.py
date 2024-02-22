from typing import List, Tuple
import json
import os
import threading

from helm.benchmark.metrics.vision_language.image_metrics import GenerateImageFromCompletionMetric, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.vision_language.image2structure.webpage.jekyll_server import JekyllServer
from helm.benchmark.scenarios.vision_language.image2structure.webpage.driver import (
    save_random_screenshot,
    ScreenshotOptions,
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

        # Convert the latex code to an image
        destination_path = os.path.join(repo_path, "screenshot.png")
        try:
            # Start the Jekyll server
            # Select a unique port per thread
            port: int = 4000 + int(threading.get_ident()) % 1000
            server = JekyllServer(repo_path, port=port)
            success: bool = server.start()
            if not success:
                # This runs on examples that are not expected to fail
                server.stop()
                raise CompilationError(f"Jekyll server failed to start: {repo_path}")

            # Take a screenshot of a random page
            try:
                scheenshot_options = self._screenshot_options
                save_random_screenshot(destination_path, port=port, options=scheenshot_options)
            except Exception as e:
                server.stop()
                raise CompilationError(f"Failed to take a screenshot: {e}")

            # Stop the server
            server.stop()
        except RuntimeError as e:
            # We do not want to catch OptionalDependencyNotInstalled (error with latex installation)
            # because it is a fatal error and should be handled by the user
            raise CompilationError from e

        # Return the image
        image: Image.Image = Image.open(destination_path)
        return image
