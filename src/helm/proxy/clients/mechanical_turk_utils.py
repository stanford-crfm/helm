import json
import re
import sys


# Source: https://github.com/charman/mturk-emoji
def replace_emoji_characters(s: str) -> str:
    """Replace 4-byte characters with HTML spans with bytes as JSON array

    This function takes a Unicode string containing 4-byte Unicode
    characters, e.g. 😀, and replaces each 4-byte character with an
    HTML span with the 4 bytes encoded as a HTML entity, e.g. &#128512;

    Args:
        s (Unicode string):
    Returns:
        Unicode string with all 4-byte Unicode characters in the source
        string replaced with HTML entities
    """

    def _emoji_match_to_span(emoji_match):
        """
        Args:
            emoji_match (MatchObject):

        Returns:
            Unicode string
        """
        return emoji_match.group().encode("ascii", "xmlcharrefreplace").decode()

    # The procedure for stripping Emoji characters is based on this
    # StackOverflow post:
    #   http://stackoverflow.com/questions/12636489/python-convert-4-byte-char-to-avoid-mysql-error-incorrect-string-value
    if sys.maxunicode == 1114111:
        # Python was built with '--enable-unicode=ucs4'
        highpoints = re.compile("[\U00010000-\U0010ffff]")
    elif sys.maxunicode == 65535:
        # Python was built with '--enable-unicode=ucs2'
        highpoints = re.compile("[\uD800-\uDBFF][\uDC00-\uDFFF]")
    else:
        raise UnicodeError("Unable to determine if Python was built using UCS-2 or UCS-4")

    return highpoints.sub(_emoji_match_to_span, s)
