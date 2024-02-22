from typing import Dict, List, Any

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


def extract_repo(base64_encoding: str, repo_path: str) -> None:
    # Decode the base64 string which corresponds to an archive
    # and extract the files to the repo_path
    try:
        archive = base64.b64decode(base64_encoding)
        # Write to .tar file
        with open(os.path.join(repo_path, "repo.tar.gz"), "wb") as f:
            f.write(archive)
        # Extract
        with tarfile.open(os.path.join(repo_path, "repo.tar.gz"), "r:gz") as tar:
            tar.extractall(repo_path)
    except Exception as e:
        raise ValueError(f"Failed to decode and extract the base64 archive: {e}")


def list_assets(repo_path: str, extensions: List[str]) -> List[str]:
    asset_paths: List[str] = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.split(".")[-1].lower() in extensions:
                # Remove repo_path from the file path
                # ignore paths that start with a dot or in _site
                if not root.startswith(os.path.join(repo_path, "_site")) and not root.startswith(
                    os.path.join(repo_path, ".")
                ):
                    asset_paths.append(os.path.relpath(os.path.join(root, file), repo_path))
    return asset_paths


def copy_assets(original_repo_path: str, destination_repo_path: str, asset_paths: List[str]) -> None:
    if not os.path.exists(original_repo_path):
        raise ValueError(f"Original repo path does not exist: {original_repo_path}")

    if original_repo_path == destination_repo_path:
        # No need to copy
        return

    if not os.path.exists(destination_repo_path):
        raise ValueError(f"Destination repo path does not exist: {destination_repo_path}")

    for asset_path in asset_paths:
        source = os.path.join(original_repo_path, asset_path)
        destination = os.path.join(destination_repo_path, asset_path)
        ensure_directory_exists(os.path.dirname(destination))
        shutil.copyfile(source, destination)


def serve_and_take_screenshot(
    repo_path: str,
    destination_path: str,
    screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    max_tries: int = 5,
) -> None:
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
    for _ in range(max_tries):
        try:
            save_random_screenshot(destination_path, port=port, options=screenshot_options)
            success = True
            break
        except Exception as e:
            if "net::ERR_CONNECTION_REFUSED" in str(e):
                error = e
                server.stop()
                time.sleep(0.5)
                server.start()
                time.sleep(0.5)
            else:
                # Do not retry
                break
    if not success:
        raise ValueError(f"Failed to take a screenshot: {error}")

    # Stop the server
    server.stop()
    time.sleep(0.1)


def compile_and_take_screenshot(
    repo_path: str,
    original_repo_path: str,
    destination_path: str,
    screenshot_options: ScreenshotOptions = ScreenshotOptions(),
) -> None:
    # Assets
    asset_paths = list_assets(repo_path, ["png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "tiff"])
    copy_assets(original_repo_path, repo_path, asset_paths)

    # Serve and take screenshot
    serve_and_take_screenshot(repo_path, destination_path, screenshot_options)


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
    ASSETS_EXTESNSIONS: List[str] = ["png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "tiff"]

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

    def preprocess_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the base64 encoding of the repo from the row and return it."""
        # Structure is a base64 encoding of the repo
        if self._output_path is None:
            raise ValueError("Output path not set")
        repo_path = os.path.join(self._output_path, f"tmp{threading.get_ident()}/{self._subset}")
        ensure_directory_exists(repo_path)

        # Decode the base64 string which corresponds to an archive
        # and extract the files to the repo_path
        structure: str = row["structure"]
        extract_repo(structure, repo_path)

        row["structure"] = repo_path
        return row

    def build_prompt(self, row: Dict[str, Any]) -> str:
        prompt: str = self.BASE_PROMPT
        asset_paths: List[str] = list_assets(row["structure"], self.ASSETS_EXTESNSIONS)
        if len(asset_paths) > 0:
            prompt += "\nYou have access to the following assets:\n"
            for asset_path in asset_paths:
                prompt += f"- {asset_path}\n"
        return prompt

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> None:
        # Structure is the path to the repo
        # Serve and take screenshot
        repo_path: str = structure
        compile_and_take_screenshot(repo_path, repo_path, destination_path, self._screenshot_options)

    def finalize(self, row: Dict[str, Any]) -> None:
        """Perform cleanup operations after the instance has been generated."""
        repo_path: str = row["structure"]
        shutil.rmtree(repo_path)
