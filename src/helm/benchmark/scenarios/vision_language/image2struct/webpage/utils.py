import re

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from html2text import HTML2Text
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


def convert_html_to_text(handler: HTML2Text, html: str) -> str:
    """Convert HTML to text

    Args:
        handler (HTML2Text): The HTML2Text handler
        html (str): The HTML to convert

    Returns:
        str: The text
    """
    text: str = handler.handle(html)
    # Normalize space sequences to a single space globally
    text = re.sub(r" +", " ", text)
    # Replace tabs with a single space
    text = re.sub(r"\t", " ", text)
    # Remove leading and trailing spaces on each line
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)
    # Remove unnecessary whitespace - multiple empty lines and tabulations
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()
