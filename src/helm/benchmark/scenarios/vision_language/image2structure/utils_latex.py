from typing import Optional, Tuple

import io
import os
import re

from helm.common.optional_dependencies import handle_module_not_found_error, OptionalDependencyNotInstalled

try:
    from latex import build_pdf
    from pdf2image import convert_from_bytes
    from PIL import ImageOps
    from PIL.Image import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2structure"])

# LaTeX preamble
# Make sure to install "latex-full".
TEX_INCLUDES = r"""
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{stfloats}
\usepackage{epstopdf}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{tikz-cd}
\usepackage{tikz-qtree}
\usepackage{tikz-dependency}
\usepackage{tikz-3dplot}
\usepackage{tikz-network}
"""

TEX_BEGIN_FILE = r"""\documentclass{article}"""
TEX_BEGIN_DOCUMENT = r"""\begin{document}"""
TEX_END_DOCUMENT = r"""\end{document}"""


def latex_to_pdf(latex_code: str, assets_path: str) -> io.BytesIO:
    # Compiling LaTeX code to PDF
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), assets_path)
    pdf = build_pdf(latex_code, texinputs=[path, ""])
    return io.BytesIO(pdf.data)  # Convert PDF to a byte stream


def pdf_to_image(
    pdf_stream: io.BytesIO,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Image:
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


def handle_latex_error(
    e: Exception,
    original_latex_code: str,
    assets_path: str,
    crop: bool,
    resize_to: Optional[Tuple[int, int]],
    num_try_remaining: int,
) -> Tuple[Image, Tuple[int, int]]:
    # Check for error that are caused by the original LaTeX code itself
    # and should not be fixed by trying again with a different code
    # TODO #2346: Make this list more exhaustive
    str_e: str = str(e).replace("\n", "")
    # Source of the descriptions:
    # - https://www.overleaf.com/learn/latex/Errors
    # - https://tex.stackexchange.com/
    for error_message in [
        # This error occurs when LaTeX encounters an undefined control sequence
        # Example: \blabla
        r"""Undefined control sequence""",
        # This error appears when you have forgotten to include an \item command.
        # It can also appear from trying to use lists inside a table incorrectly.
        # Example:
        #     \begin{itemize}
        #     First item without the \item command
        #     \end{itemize}
        r"""LaTeX Error: Lonely \item--perhaps a missing list environment.""",
        # This error occurs when a { or } is missing.
        # Example: \sum_{i=1 ^n
        r"""Missing } inserted""",
        r"""Missing { inserted""",
        # This error occurs when LaTeX encounters a double subscript.
        # Example: a_b_c
        r"""Double subscript.""",
        # This error occurs when an environment or $ is added around something that cannot be typeset
        # in the given mode.
        # Example:
        #      $
        #      \begin{table}
        #      ...
        #      \end{table}
        #      $
        r"""LaTeX Error: Not in outer par mode.""",
        # This error occurs when LaTeX is typesetting a table and detects
        # an alignment character ( & ) where it did not expect to find one
        r"""Extra alignment tab has been changed to \cr.""",
        # Missing control sequence othen than $ (which is handled elsewhere).
        # Example: \left( without
        "Missing \\",
        # LaTeX Error: \begin{<env>} on input line <line> ended by \end{<diff_env>}
        # This error occurs when LaTeX encounters an environment that is not properly closed.
        # Example:
        #     \begin{table}
        #     ...
        #     \end{document}
        r"""LaTeX Error: \begin{""",
        # This error occurs when LaTeX encounters a \noalign command in the wrong place.
        # Example:
        #     \begin{tabular}
        #     \noalign{\hrule}
        #     ...
        #     \end{tabular}
        r"""Misplaced \noalign""",
    ]:
        if error_message in str_e:
            raise RuntimeError("Undefined control sequence in LaTeX code") from e

    if num_try_remaining > 0:
        # Check if the error is easily fixable
        fixed_code: str = original_latex_code

        # Equation not in math mode
        # We correct this error as the prompt might not be obvious if the output should be:
        # <EQUATION_CODE> or $<EQUATION_CODE>$.
        # We only handle this cas and that is why we add the $ at the beginning and end of the equation.
        # The missing $ might come from elsewhere but then, it is a problem of the generated code,
        # and not some unclear instructions, so we do not handle it.
        # Error format: "Missing $ inserted" or "<command> allowed only in math mode"
        if "Missing $ inserted" in str(e) or " allowed only in math mode" in str_e:
            fixed_code = f"${fixed_code}$"

        # Missing include
        # Missing includes are tolerated as the prompt suggests that it is not necessary to include them,
        # and our TEX_INCLUDES might lack some packages.
        # Error format: "LaTeX Error: Environment <env> undefined."
        undefined_seach = re.search(r"LaTeX Error: Environment (.*) undefined", str_e)
        if undefined_seach:
            env_undefined: str = undefined_seach.group(1)
            index_insertion = fixed_code.find(TEX_BEGIN_DOCUMENT)
            if index_insertion != -1:
                raise RuntimeError("Could not add missing include") from e
            fixed_code = (
                fixed_code[:index_insertion] + f"\\usepackage{{{env_undefined}}}\n" + fixed_code[index_insertion:]
            )

        # Try again with the fixed code (if the fixed code is different from the original code)
        if fixed_code != original_latex_code:
            return latex_to_image(
                fixed_code,
                assets_path=assets_path,
                crop=crop,
                resize_to=resize_to,
                num_try_remaining=num_try_remaining - 1,
            )

    # TODO #2346: Ideally we should never reach this point
    # All errors should be either detected as:
    # - generation error: should not be fixed and raised
    # - easily fixable: should be fixed and tried again
    # If we reach this point, it means that none of the above cases were detected.
    print("\n\n\nOriginal LaTeX code:", original_latex_code)
    raise Exception("Error while processing LaTeX code") from e
    raise RuntimeError from e


def latex_to_image(
    original_latex_code: str,
    assets_path: str,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
    num_try_remaining: int = 1,
) -> Tuple[Image, Tuple[int, int]]:
    # Basic LaTeX processing
    # This changes cannot break the original LaTeX code
    # Other processing will be done in the handle_latex_error function
    # but these might break the original LaTeX code so they are only applied
    # if the original LaTeX code does not compile.

    # 1. Add begin/end document if not present
    latex_code: str = original_latex_code
    if TEX_BEGIN_DOCUMENT not in latex_code and TEX_BEGIN_FILE not in latex_code:
        latex_code = TEX_BEGIN_DOCUMENT + latex_code
    if TEX_END_DOCUMENT not in latex_code:
        latex_code = latex_code + TEX_END_DOCUMENT
    # 2. Add preamble
    # Even if the original latex code already has some includes,
    # we are going to add ours (including twice the same package is not a problem)
    # and replace whatever \documentclass is present with our own
    # 2.2. Remove \documentclass if present
    documentclass_search = re.search(r"\\documentclass\{(.*)\}", latex_code)
    if documentclass_search:
        documentclass: str = documentclass_search.group(1)
        latex_code = latex_code.replace(f"\\documentclass{{{documentclass}}}", "")
    latex_code = f"{TEX_BEGIN_FILE}\n{TEX_INCLUDES}\n{latex_code}"

    try:
        pdf_stream = latex_to_pdf(latex_code, assets_path=assets_path)
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
            return handle_latex_error(e, original_latex_code, assets_path, crop, resize_to, num_try_remaining)
    except Exception as e:
        return handle_latex_error(e, original_latex_code, assets_path, crop, resize_to, num_try_remaining)
