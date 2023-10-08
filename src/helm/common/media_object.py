import os
import urllib
from copy import deepcopy
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class MultimediaObject:
    """Represents a sequence of `MediaObject`s."""

    content: List[MediaObject] = field(default_factory=list)
    """The sequence of `MediaObject`s."""

    def add_textual_prefix(self, prefix: str) -> "MultimediaObject":
        """
        Add a prefix to the beginning of the multimodal sequence.
        :param prefix: The prefix to add.
        :return: New multimodal object with prefix.
        """
        result: MultimediaObject = deepcopy(self)
        if not prefix:
            return result

        start: MediaObject = result.content[0]
        if start.type == "text":
            result.content[0] = replace(result.content[0], text=prefix + start.text)
        else:
            result.content.insert(0, MediaObject(text=prefix, content_type="text/plain"))
        return result

    def add_textual_suffix(self, suffix: str) -> "MultimediaObject":
        """
        Add a suffix to the end of the multimodal sequence.
        :param suffix: The suffix to add.
        :return: New multimodal content with suffix.
        """
        result: MultimediaObject = deepcopy(self)
        if not suffix:
            return result

        end: MediaObject = result.content[-1]
        if end.type == "text":
            result.content[-1] = replace(result.content[-1], text=end.text + suffix)
        else:
            result.content.append(MediaObject(text=suffix, content_type="text/plain"))
        return result

    def combine(self, other: "MultimediaObject") -> "MultimediaObject":
        """
        Combine this multimodal content with another multimodal content.
        :param other: The other multimodal content.
        :return: The combined multimodal content.
        """
        return MultimediaObject(content=self.content + other.content)

    @property
    def text(self) -> str:
        """
        Get the text-only part of this multimodal content.
        :return: The text-only representation.
        """
        return "".join(item.text for item in self.content if item.text)
