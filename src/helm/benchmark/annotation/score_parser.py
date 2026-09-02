"""Helpers for parsing structured values from model-as-judge responses."""

import re
from typing import Match, Optional, Pattern, Union


def find_last_match(pattern: Union[str, Pattern[str]], text: str, flags: int = 0) -> Optional[Match[str]]:
    """Return the final match in a response, or ``None`` when absent.

    Judge responses can quote the evaluated model before emitting their own
    verdict. Using the final structured value prevents an earlier quoted value
    from being mistaken for the judge's answer.
    """
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def find_last_tagged_match(text: str, tag: str) -> Optional[Match[str]]:
    """Return the final fuzzy-matched XML-like tag in a response."""
    pattern = rf"<\s*{re.escape(tag)}\s*>(.*?)<\/?\s*{re.escape(tag)}\s*>"
    return find_last_match(pattern, text, re.DOTALL | re.IGNORECASE)
