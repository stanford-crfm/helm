from typing import Tuple, Dict, Any
import os
import subprocess
import tempfile

from helm.benchmark.annotation.image2struct.image_compiler_annotator import ImageCompilerAnnotator, CompilationError
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error, OptionalDependencyNotInstalled

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError as ex:
    handle_module_not_found_error(ex, suggestions=["images"])


class LilypondCompilerAnnotator(ImageCompilerAnnotator):
    """Annotator that compiles the text completions into a music sheet with LilyPond."""

    name: str = "lilypond_compiler"
    base_path = "lilypond-2.24.3/bin"

    def __init__(self, cache_config: CacheConfig, file_storage_path: str):
        super().__init__(cache_config, file_storage_path)
        try:
            result = subprocess.run([f"{self.base_path}/lilypond", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise OptionalDependencyNotInstalled(
                    "LilyPond is not installed. Download and install it from https://lilypond.org/download.html"
                )
        except FileNotFoundError as e:
            raise OptionalDependencyNotInstalled(
                "LilyPond is not installed. Download and install it from https://lilypond.org/download.html.\n"
                f"Original error: {e}"
            ) from e

    def compile_completion_into_image(
        self, request_state: RequestState, completion_text: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Given a completion, compile it with LilyPond."""
        # The LilyPond command requires a file on disk, so we write the completion to a temporary file
        tmp = tempfile.NamedTemporaryFile()
        ly_file_path: str = f"{tmp.name}.ly"
        with open(ly_file_path, "w") as f:
            f.write(completion_text)

        # What we pass in as -o should be the same name as the .ly file, but without the extension
        output_path: str = ly_file_path.replace(".ly", "")
        # The image file of the music sheet should have the same path as the .ly file, but with .png extension
        sheet_music_path: str = ly_file_path.replace(".ly", ".png")

        try:
            # Edits the LilyPond file to be compatible with the current version
            result = subprocess.run(
                [f"{self.base_path}/convert-ly", "-e", ly_file_path], capture_output=True, text=True
            )
            assert result.returncode == 0, f"convert-ly failed: {result.stderr}"

            # Generate PNG image from the LilyPond file
            # LilyPond supports partial compilation, which means it attempts to produce an image
            # for the correct portions of the code, even if there are errors elsewhere
            subprocess.run(
                [f"{self.base_path}/lilypond", "--png", "-o", output_path, ly_file_path], capture_output=True, text=True
            )
            # If an image file is not generated, we consider it an absolute compilation failure
            assert os.path.exists(sheet_music_path), "lilypond did not generate the image"

            # Load the image as a PIL Image object
            image = Image.open(sheet_music_path)

            # Crop the image to remove the white space around the music sheet
            (w, h) = image.size
            # Remove pagination
            image = image.crop((0, 0, w, h - int(h * 0.2)))  # type: ignore
            # Remove white border
            image = image.crop(ImageOps.invert(image).getbbox())  # type: ignore
        except (AssertionError, RuntimeError) as e:
            raise CompilationError(str(e)) from e
        finally:
            # Clean up the temporary files
            if os.path.exists(ly_file_path):
                os.remove(ly_file_path)
            if os.path.exists(sheet_music_path):
                os.remove(sheet_music_path)

        return image, dict()
