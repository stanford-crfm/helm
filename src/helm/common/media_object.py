import os
import urllib
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import List


@dataclass(frozen=True)
class MediaObject:
    """A media object e.g., image, video, audio, etc."""

    content_type: str
    """
    A valid Multipurpose Internet Mail Extensions (MIME) type for this media object in the format `<type>/<subtype>`.
    IANA is the official registry of MIME media types and maintains a list of all the official MIME types:
    https://www.iana.org/assignments/media-types/media-types.xhtml
    Some common MIME types can be found here:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    """

    text: str = ""
    """When the media object is text."""

    location: str = ""
    """When the media object is a file, specify the location of the media object, which can be a local path or URL."""

    @property
    def type(self) -> str:
        """The MIME type of the media object."""
        return self.content_type.split("/")[0]

    @property
    def subtype(self) -> str:
        """The MIME subtype of the media object."""
        return self.content_type.split("/")[1]

    @property
    def is_local_file(self) -> bool:
        """Returns `True` if the media object is a local file and False if `location` is a URL."""
        return urllib.parse.urlparse(self.location).scheme not in ["http", "https"]

    def __post_init__(self):
        """Validates the media object field values."""
        # Verify that the `mime_type` is in the correct format
        assert len(self.content_type.split("/")) == 2, f"Invalid MIME type: {self.content_type}"

        if self.location and self.is_local_file:
            # Checks that the `location` is a valid local file path
            assert os.path.exists(self.location), f"Local file does not exist at path: {self.location}"


# Helper functions for `MediaObject`s
def add_textual_prefix(multimodal_content: List[MediaObject], prefix: str) -> List[MediaObject]:
    """
    Add a prefix to the beginning of the multimodal sequence made up of `MediaObject`s.

    :param multimodal_content: The multimodal sequence of `MediaObject`s
    :param prefix: The prefix to add.
    :return: a list of `MediaObject`s with prefix.
    """
    result: List[MediaObject] = deepcopy(multimodal_content)
    if not prefix:
        return result

    start: MediaObject = result[0]
    if start.type == "text":
        result[0] = replace(result[0], text=prefix + start.text)
    else:
        result.insert(0, MediaObject(text=prefix, content_type="text/plain"))
    return result


def add_textual_suffix(multimodal_content: List[MediaObject], suffix: str) -> List[MediaObject]:
    """
    Add a suffix to the end of the multimodal sequence made up of `MediaObject`s.

    :param multimodal_content: The multimodal sequence of `MediaObject`s
    :param suffix: The suffix to add.
    :return: a list of `MediaObject`s with suffix.
    """
    result: List[MediaObject] = deepcopy(multimodal_content)
    if not suffix:
        return result

    end: MediaObject = result[-1]
    if end.type == "text":
        result[-1] = replace(result[-1], text=end.text + suffix)
    else:
        result.append(MediaObject(text=suffix, content_type="text/plain"))
    return result


def extract_text(multimodal_content: List[MediaObject]) -> str:
    """
    Extracts the text from a multimodal sequence of `MediaObject`s.

    :param multimodal_content: The multimodal sequence of `MediaObject`s
    :return: The text extracted from the multimodal sequence.
    """
    return "".join([m.text for m in multimodal_content if m.type == "text"])
