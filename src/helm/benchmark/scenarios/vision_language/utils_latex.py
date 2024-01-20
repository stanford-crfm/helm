from typing import Optional, Tuple

from latex import build_pdf
from pdf2image import convert_from_bytes
from PIL import ImageOps
import io
import os


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
    pdf_stream = latex_to_pdf(latex_code, assets_path=assets_path)
    image = pdf_to_image(pdf_stream, crop=crop, resize_to=resize_to)
    return image, image.size
