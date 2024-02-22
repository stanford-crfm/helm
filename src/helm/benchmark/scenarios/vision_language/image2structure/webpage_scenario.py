from helm.benchmark.scenarios.scenario import VALID_SPLIT
from helm.benchmark.scenarios.vision_language.image2structure.image2structure_scenario import Image2StructureScenario
from helm.benchmark.scenarios.vision_language.image2structure.webpage.jekyll_server import JekyllServer
from helm.benchmark.scenarios.vision_language.image2structure.webpage.driver import (
    save_random_screenshot,
    ScreenshotOptions,
)
from helm.common.general import ensure_directory_exists

import base64
import os
import threading
import tarfile
import shutil
import time


class WebpageScenario(Image2StructureScenario):
    BASE_PROMPT = (
        "Please provide the source code used to generate this webpage."
        " You should output a json object associating each file name with its content.\n\n"
        "Here is a simple example (that does not correspond to the image):\n"
        "[\n"
        "  {\n"
        '    "filename": "index.html",\n'
        '    "content": "<!DOCTYPE html>\\n<html>\\n<head>\\n<title>Title of the document</title>\\n</head>\\n<body>\\n\\n<p>Content of the document......</p>\\n\\n</body>\\n</html>"\n'  # noqa: E501
        "  },\n"
        "  {\n"
        '    "filename": "style.css",\n'
        '    "content": "body {\\n  background-color: lightblue;\\n}\\nh1 {\\n  color: white;\\n  text-align: center;\\n}"\n'  # noqa: E501
        "  },\n"
        "  {\n"
        '    "filename": "script.js",\n'
        '    "content": "document.getElementById(\\"demo\\").innerHTML = \\"Hello JavaScript!\\";"\n'
        "  }\n"
        "]\n"
        "\nNow provide the source code for the webpage provided as an image."
    )

    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-webpage"
    SUBSETS = ["css", "html", "javascript"]
    MAX_TRIES: int = 5

    name = "image2webpage"
    description = "Evaluate multimodal models on webpage generation to recreate a provided image"

    def __init__(
        self,
        subset: str,
        recompile_prompt: bool = True,
        split: str = VALID_SPLIT,
        screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    ):
        super().__init__(subset, recompile_prompt, split)
        self._screenshot_options = screenshot_options

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> None:
        # Structure is a base64 encoding of the repo
        if self._output_path is None:
            raise ValueError("Output path not set")
        repo_path = os.path.join(self._output_path, f"tmp{threading.get_ident()}/{self._subset}")
        ensure_directory_exists(repo_path)

        # Decode the base64 string which corresponds to an archive
        # and extract the files to the repo_path
        try:
            archive = base64.b64decode(structure)
            # Write to .tar file
            with open(os.path.join(repo_path, "repo.tar.gz"), "wb") as f:
                f.write(archive)
            # Extract
            with tarfile.open(os.path.join(repo_path, "repo.tar.gz"), "r:gz") as tar:
                tar.extractall(repo_path)
        except Exception as e:
            raise ValueError(f"Failed to decode and extract the base64 archive: {e}")

        # Start the Jekyll server
        # Select a unique port per thread
        port: int = 4000 + int(threading.get_ident()) % 1000
        server = JekyllServer(repo_path, port=port)
        success: bool = server.start()
        if not success:
            # This runs on examples that are not expected to fail
            server.stop()
            raise ValueError(f"Jekyll server failed to start: {repo_path}")

        # Take a screenshot of a random page
        success = False
        error: Exception
        for _ in range(self.MAX_TRIES):
            try:
                scheenshot_options = self._screenshot_options
                save_random_screenshot(destination_path, port=port, options=scheenshot_options)
                success = True
                break
            except Exception as e:
                error = e
                server.stop()
                time.sleep(0.5)
                server.start()
                time.sleep(0.5)
        if not success:
            raise ValueError(f"Failed to take a screenshot: {error}")

        # Stop the server
        shutil.rmtree(repo_path)
        server.stop()
        time.sleep(0.1)
