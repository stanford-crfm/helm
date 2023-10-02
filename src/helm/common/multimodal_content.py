from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Union

from .file import File


@dataclass(frozen=True)
class MultimodalContent:
    content: List[Union[str, File]] = field(default_factory=list)
    """Multimodal content can consist of text and files (e.g., images, audio files, etc.)"""

    def add_textual_prefix(self, prefix: str) -> "MultimodalContent":
        """
        Add a prefix to the beginning of the multimodal sequence.

        :param prefix: The prefix to add.
        :return: New multimodal content with prefix.
        """
        result: MultimodalContent = deepcopy(self)
        if not prefix:
            return result

        start: Union[str, File] = result.content[0]
        if isinstance(start, File):
            result.content.insert(0, prefix)
        else:
            result.content[0] = prefix + start
        return result

    def add_textual_suffix(self, suffix: str) -> "MultimodalContent":
        """
        Add a suffix to the end of the multimodal sequence.

        :param suffix: The suffix to add.
        :return: New multimodal content with suffix.
        """
        result: MultimodalContent = deepcopy(self)
        if not suffix:
            return result

        end: Union[str, File] = result.content[-1]
        if isinstance(end, File):
            result.content.append(suffix)
        else:
            result.content[-1] = end + suffix
        return result

    def combine(self, other: "MultimodalContent") -> "MultimodalContent":
        """
        Combine this multimodal content with another multimodal content.

        :param other: The other multimodal content.
        :return: The combined multimodal content.
        """
        return MultimodalContent(content=self.content + other.content)

    @property
    def text(self) -> str:
        """
        Get the text-only part of this multimodal content.

        :return: The text-only representation.
        """
        return "".join(item for item in self.content if isinstance(item, str))

    def __str__(self) -> str:
        return "".join([content.location if isinstance(content, File) else content for content in self.content])
