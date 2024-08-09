from typing import Dict, List, Any, Optional

from helm.benchmark.annotation.image2struct.image_compiler_annotator import CompilationError
from helm.benchmark.scenarios.scenario import VALID_SPLIT
from helm.benchmark.scenarios.vision_language.image2struct.image2struct_scenario import (
    Image2StructureScenario,
    PROCESSED,
    DIFFICULTY_ALL,
)
from helm.benchmark.scenarios.vision_language.image2struct.webpage.jekyll_server import JekyllServer
from helm.benchmark.scenarios.vision_language.image2struct.webpage.driver import (
    save_random_screenshot,
    ScreenshotOptions,
)
from helm.benchmark.scenarios.vision_language.image2struct.webpage.utils import convert_html_to_text
from helm.common.general import ensure_directory_exists
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.hierarchical_logger import hlog

try:
    from html2text import HTML2Text
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


import base64
import os
import threading
import tarfile
import shutil
import time
import pickle


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


def serve_and_take_screenshot(
    repo_path: str,
    destination_path: str,
    screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    max_tries: int = 5,
) -> Dict[str, Any]:
    # Start the Jekyll server
    # Select a unique port per thread
    port: int = 4000 + int(threading.get_ident()) % 1000
    server = JekyllServer(repo_path, port=port, verbose=False)
    success: bool = server.start()
    if not success:
        # This runs on examples that are not expected to fail
        server.stop()
        hlog(f"Failed to start the Jekyll server: {repo_path} on port {port}. Will raise a ValueError.")
        raise ValueError(f"Jekyll server failed to start: {repo_path}")

    # Take a screenshot of a random page
    success = False
    error: Optional[Exception] = None

    MAX_TRIES_ALL_ERRORS = 3
    MAX_TRIES_CONNECTION_REFUSED = 5
    MAX_TRIES = max(MAX_TRIES_ALL_ERRORS, MAX_TRIES_CONNECTION_REFUSED)
    for compilation_attempt in range(MAX_TRIES):
        try:
            infos: Dict[str, Any] = save_random_screenshot(destination_path, port=port, options=screenshot_options)
            success = True
            break
        except Exception as e:
            error = e

            if "net::ERR_CONNECTION_REFUSED" in str(e) and compilation_attempt < MAX_TRIES_CONNECTION_REFUSED:
                hlog(
                    f"Failed to take a screenshot: ERR_CONNECTION_REFUSED [Attempt {compilation_attempt + 1}/"
                    f"{MAX_TRIES_CONNECTION_REFUSED}]. Error: {e}. Retrying..."
                )
                server.stop()
                time.sleep(0.5)
                server.start()
                time.sleep(0.5)
            elif compilation_attempt < MAX_TRIES_ALL_ERRORS:
                hlog(
                    f"Failed to take a screenshot: Unknown [Attempt {compilation_attempt + 1}/{MAX_TRIES_ALL_ERRORS}]."
                    f" Error: {e}. Retrying..."
                )
            else:
                # Do not retry
                hlog(
                    f"Failed to take a screenshot: Unknown [Attempt {compilation_attempt + 1}/{MAX_TRIES_ALL_ERRORS}]."
                    f" Error: {e}. Raising CompilationError."
                )
                break

    if not success:
        raise CompilationError(f"Failed to take a screenshot: {error}")

    # Stop the server
    server.stop()
    time.sleep(0.1)

    return infos


class WebpageScenario(Image2StructureScenario):
    BASE_PROMPT = (
        "Please generate the source code to generate a webpage that looks like this image as much as feasibly possible.\n"  # noqa: E501
        "You should output a json object associating each file name with its content.\n\n"
        "Here is a simple example of the expected structure (that does not correspond to the image)."
        " In this example, 3 files are created: index.html, style.css and script.js.\n"
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
        "You do not have to create files with the same names. Create as many files as you need, you can even use directories if necessary,"  # noqa: E501
        " they will be created for you automatically. Try to write some realistic code keeping in mind that it should"
        " look like the image as much as feasibly possible."
    )

    HUGGINGFACE_DATASET_NAME = "stanford-crfm/i2s-webpage"
    SUBSETS = ["css", "html", "javascript", "wild", "wild_legacy"]
    MAX_TRIES: int = 5
    ASSETS_EXTENSIONS: List[str] = ["png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "tiff"]

    name = "image2webpage"
    description = "Evaluate multimodal models on webpage generation to recreate a provided image"

    def __init__(
        self,
        subset: str,
        recompile_prompt: bool = True,
        split: str = VALID_SPLIT,
        difficulty: str = DIFFICULTY_ALL,
        screenshot_options: ScreenshotOptions = ScreenshotOptions(),
    ):
        super().__init__(subset, recompile_prompt, split, difficulty)
        self._screenshot_options = screenshot_options
        self._html2text = HTML2Text()
        self._html2text.ignore_links = True

    def preprocess_row(self, row: Dict[str, Any], assets_path: str) -> Dict[str, Any]:
        """Extract the base64 encoding of the repo from the row and return it."""
        # No need to reprocess if the assets are already saved
        assets_save_path: str = os.path.join(assets_path, str(row["uuid"].replace('"', "")))
        if os.path.exists(assets_save_path):
            try:
                with open(os.path.join(assets_save_path, "assets_paths.pkl"), "rb") as f:
                    row["assets_paths"] = pickle.load(f)
                with open(os.path.join(assets_save_path, "assets_names.pkl"), "rb") as f:
                    row["assets_names"] = pickle.load(f)
                del row["assets"]
                row["structure"] = PROCESSED
                return row
            except Exception:
                # There was an issue when loading the assets, reprocess
                shutil.rmtree(assets_save_path)
        ensure_directory_exists(assets_save_path)

        if "wild" in self._subset:
            # There is no stucture
            del row["assets"]
            row["assets_paths"] = []
            row["assets_names"] = []
            return row

        # Structure is a base64 encoding of the repo
        if self._output_path is None:
            raise ValueError("Output path not set")
        repo_path = os.path.join(self._output_path, f"tmp{threading.get_ident()}_{self._subset}")
        ensure_directory_exists(repo_path)

        # Decode the base64 string which corresponds to an archive
        # and extract the files to the repo_path
        structure: str = row["structure"]
        extract_repo(structure, repo_path)
        row["structure"] = PROCESSED
        row["repo_path"] = repo_path  # Stored for cleanup

        # Process the assets
        asset_paths: List[str] = list_assets(repo_path, self.ASSETS_EXTENSIONS)
        del row["assets"]
        row["assets_paths"] = []
        row["assets_names"] = []
        # Copy each asset to a unique persistent path
        for i, asset_local_path in enumerate(asset_paths):
            asset_name: str = asset_local_path
            asset_dest_path = os.path.join(assets_save_path, f"{i}.{asset_local_path.split('.')[-1]}")
            shutil.copyfile(os.path.join(row["repo_path"], asset_local_path), asset_dest_path)
            row["assets_paths"].append(asset_dest_path)
            row["assets_names"].append(asset_name)

        # Save both assets_paths and assets_names as files than can be loaded
        with open(os.path.join(assets_save_path, "assets_paths.pkl"), "wb") as f:
            pickle.dump(row["assets_paths"], f)
        with open(os.path.join(assets_save_path, "assets_names.pkl"), "wb") as f:
            pickle.dump(row["assets_names"], f)

        return row

    def build_prompt(self, row: Dict[str, Any]) -> str:
        prompt: str = self.BASE_PROMPT
        assert "assets_paths" in row, "No assets paths in the row"
        assert "assets_names" in row, "No assets names in the row"
        assert len(row["assets_paths"]) == len(row["assets_names"])
        if len(row["assets_names"]) > 0:
            prompt += "\nYou have access to the following assets:\n"
            for asset_local_path in row["assets_names"]:
                prompt += f"- {asset_local_path}\n"
        return prompt

    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> str:
        # Structure is the path to the repo
        # Serve and take screenshot
        repo_path: str = structure
        infos: Dict[str, Any] = serve_and_take_screenshot(repo_path, destination_path, self._screenshot_options)
        text: str = convert_html_to_text(self._html2text, infos["html"])
        return text

    def finalize(self, row: Dict[str, Any]) -> None:
        """Perform cleanup operations after the instance has been generated."""
        if "repo_path" in row:
            repo_path: str = row["repo_path"]
            shutil.rmtree(repo_path)
