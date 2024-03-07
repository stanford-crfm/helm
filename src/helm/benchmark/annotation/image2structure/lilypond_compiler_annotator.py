from typing import Tuple, Dict, Any
import os
import subprocess
import tempfile

from helm.benchmark.annotation.image2structure.image_compiler_annotator import ImageCompilerAnnotator, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from PIL import Image
except ModuleNotFoundError as ex:
    handle_module_not_found_error(ex, suggestions=["images"])


class LilyPondAnnotator(ImageCompilerAnnotator):
    """Annotator that compiles the text completions into a music sheet with LilyPond."""

    name: str = "lilypond_compiler"

    def __int__(self, cache_config: CacheConfig, file_storage_path: str):
        super().__init__(cache_config, file_storage_path)
        result = subprocess.run(["lilypond", "--version"], capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), "LilyPond is not installed. Download and install it from https://lilypond.org/download.html"

    def compile_completion_into_image(
        self, request_state: RequestState, completion_text: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Given a completion, compile it with LilyPond."""
        # The LilyPond command requires a file on disk, so we write the completion to a temporary file
        tmp = tempfile.NamedTemporaryFile()
        ly_file_path: str = f"{tmp.name}.ly"
        with open(ly_file_path, "w") as f:
            f.write(completion_text)

        # The image file of the music sheet should have the same path as the .ly file, but with .png extension
        sheet_music_path: str = ly_file_path.replace(".ly", ".png")

        try:
            # Edits the LilyPond file to be compatible with the current version
            result = subprocess.run(["convert-ly", "-e", ly_file_path], capture_output=True, text=True)
            assert result.returncode == 0, f"convert-ly failed: {result.stderr}"

            # Generate PNG image from the LilyPond file
            result = subprocess.run(["lilypond", "--png", ly_file_path], capture_output=True, text=True)
            assert result.returncode == 0, f"lilypond failed: {result.stderr}"
            assert os.path.exists(sheet_music_path), "lilypond did not generate the image"

            # Load the image as a PIL Image object
            image = Image.open(sheet_music_path)
        except (AssertionError, RuntimeError) as e:
            raise CompilationError(str(e)) from e
        finally:
            # Clean up the temporary files
            if os.path.exists(ly_file_path):
                os.remove(ly_file_path)
            if os.path.exists(sheet_music_path):
                os.remove(sheet_music_path)

        return image, dict()
