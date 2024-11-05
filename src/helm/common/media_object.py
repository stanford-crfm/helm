import os
import urllib
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


IMAGE_TYPE = "image"
TEXT_TYPE = "text"


@dataclass(frozen=True)
class MediaObject:
    """A media object e.g., image, video, audio, etc."""

    content_type: str
    """A valid Multipurpose Internet Mail Extensions (MIME) type for this media object in the format `<type>/<subtype>`.
    IANA is the official registry of MIME media types and maintains a list of all the official MIME types:
    https://www.iana.org/assignments/media-types/media-types.xhtml
    Some common MIME types can be found here:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    """

    text: Optional[str] = None
    """When the `type` of media object is text."""

    location: Optional[str] = None
    """When the media object is a file, specify the location of the media object, which can be a local path or URL."""

    def to_dict(self) -> Dict[str, Any]:
        """Converts the media object to a dictionary."""
        return {key: value for key, value in self.__dict__.items() if value is not None}

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
        if self.location is None:
            return False
        return urllib.parse.urlparse(self.location).scheme not in ["http", "https"]

    def __post_init__(self):
        """Validates the media object field values."""
        # Verify that the `mime_type` is in the correct format
        assert len(self.content_type.split("/")) == 2, f"Invalid MIME type: {self.content_type}"

        if self.type == TEXT_TYPE:
            assert self.text is not None
            assert self.location is None
        else:
            assert self.text is None
            assert self.location is not None

            # Checks that the `location` is a valid local file path
            if self.is_local_file:
                assert os.path.exists(self.location), f"Local file does not exist at path: {self.location}"

    def is_type(self, media_type: str) -> bool:
        """Returns `True` if the media object is of the specified type."""
        return self.type == media_type


@dataclass(frozen=True)
class MultimediaObject:
    """Represents a sequence of `MediaObject`s."""

    media_objects: List[MediaObject] = field(default_factory=list)
    """The sequence of `MediaObject`s."""

    def add_textual_prefix(self, prefix: str) -> "MultimediaObject":
        """
        Returns a new `MultimediaObject` with a textual prefix added to the beginning of the multimodal sequence.
        :param prefix: The prefix to add.
        :return: New multimodal object with prefix.
        """
        result: MultimediaObject = deepcopy(self)
        if not prefix:
            return result

        start: MediaObject = result.media_objects[0]
        if start.is_type(TEXT_TYPE) and start.text:
            result.media_objects[0] = replace(result.media_objects[0], text=prefix + start.text)
        else:
            result.media_objects.insert(0, MediaObject(text=prefix, content_type="text/plain"))
        return result

    def add_textual_suffix(self, suffix: str) -> "MultimediaObject":
        """
        Returns a new `MultimediaObject` with a textual suffix added to the end of the multimodal sequence.
        :param suffix: The suffix to add.
        :return: New multimodal content with suffix.
        """
        result: MultimediaObject = deepcopy(self)
        if not suffix:
            return result

        end: MediaObject = result.media_objects[-1]
        if end.is_type(TEXT_TYPE) and end.text:
            result.media_objects[-1] = replace(result.media_objects[-1], text=end.text + suffix)
        else:
            result.media_objects.append(MediaObject(text=suffix, content_type="text/plain"))
        return result

    def combine(self, other: "MultimediaObject") -> "MultimediaObject":
        """
        Return a new `MultimediaObject` that contains the contents of this object and the other object.
        :param other: The other multimodal content.
        :return: The combined multimodal content.
        """
        return MultimediaObject(media_objects=self.media_objects + other.media_objects)

    @property
    def size(self) -> int:
        """
        Get the number of `MediaObject`s in this multimodal content.
        :return: The number of `MediaObject`s .
        """
        return len(self.media_objects)

    @property
    def text(self) -> str:
        """
        Get the text-only part of this multimodal content.
        :return: The text-only representation.
        """
        return "".join(item.text for item in self.media_objects if item.is_type(TEXT_TYPE) and item.text)
