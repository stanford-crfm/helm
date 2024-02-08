from typing import Optional, Tuple

import io
import os

from helm.common.optional_dependencies import handle_module_not_found_error, OptionalDependencyNotInstalled

try:
    from latex import build_pdf
    from pdf2image import convert_from_bytes
    from PIL import ImageOps
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])

# LaTeX preamble
# Make sure to install "latex-full".
TEX_BEGIN = r"""
\documentclass{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{stfloats}
\usepackage{epstopdf}
\usepackage{pgfplots}
\begin{document}"""

TEX_END = r"""\end{document}"""


def latex_to_pdf(latex_code: str, assets_path: str) -> io.BytesIO:
    # Compiling LaTeX code to PDF
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), assets_path)
    pdf = build_pdf(latex_code, texinputs=[path, ""])
    return io.BytesIO(pdf.data)  # Convert PDF to a byte stream


def pdf_to_image(
    pdf_stream: io.BytesIO,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
):  # -> Image
    # Convert the first page of the PDF stream to an image
    images = convert_from_bytes(pdf_stream.read(), first_page=1, last_page=1)
    if images:
        image = images[0]

        # Removes the white border around the image
        if crop:
            (w, h) = image.size
            image = image.crop((0, 0, w, h - int(h * 0.13)))  # Remove pagination
            image = image.crop(ImageOps.invert(image).getbbox())  # Remove white border

        # Resize the image
        if resize_to:
            image = image.resize(resize_to)

        return image
    else:
        raise Exception("PDF to Image conversion failed")


def latex_to_image(
    latex_code: str,
    assets_path: str,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
):  # -> Tuple[Image, Tuple[int, int]]:
    try:
        pdf_stream = latex_to_pdf(TEX_BEGIN + latex_code + TEX_END, assets_path=assets_path)
        image = pdf_to_image(pdf_stream, crop=crop, resize_to=resize_to)
        return image, image.size
    except RuntimeError as e:
        if str(e) == "No available builder could be instantiated. Please make sure LaTeX is installed.":
            raise OptionalDependencyNotInstalled(
                "Optional dependency LaTeX is not installed. "
                "Please install LaTeX and make sure it is available in your PATH."
                "You can install LaTeX on Ubuntu with `sudo apt-get install texlive-full`."
            ) from e
        else:
            raise e
