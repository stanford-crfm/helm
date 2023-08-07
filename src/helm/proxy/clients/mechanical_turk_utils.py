import codecs
import json
import re
import sys


# Source: https://github.com/charman/mturk-emoji
def replace_emoji_characters(s):
    """Replace 4-byte characters with HTML spans with bytes as JSON array

    This function takes a Unicode string containing 4-byte Unicode
    characters, e.g. 😀, and replaces each 4-byte character with an
    HTML span with the 4 bytes encoded as a JSON array, e.g.:

      <span class='emoji-bytes' data-emoji-bytes='[240, 159, 152, 128]'></span>

    Args:
        s (Unicode string):
    Returns:
        Unicode string with all 4-byte Unicode characters in the source
        string replaced with HTML spans
    """

    def _emoji_match_to_span(emoji_match):
        """
        Args:
            emoji_match (MatchObject):

        Returns:
            Unicode string
        """
        bytes = codecs.encode(emoji_match.group(), "utf-8")
        bytes_as_json = json.dumps([b for b in bytearray(bytes)])
        return "<span class='emoji-bytes' data-emoji-bytes='%s'></span>" % bytes_as_json

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
