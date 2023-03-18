import json

from dataclasses import dataclass
from typing import Generator


@dataclass(frozen=True, eq=False)
class DocumentReadingProcessor:
    """
    TODO: This is probably a bad class name.
    """

    file_path: str
    """The path to the data file"""

    file_format: str
    """The type of the file format, which determines how it will be read"""

    def get_document_generator(self) -> Generator:
        if self.file_format == "the_pile":
            return self.get_the_pile_document_generator()
        elif self.file_format == "raw":
            return self.get_raw_document_generator()
        elif self.file_format == "custom":
            return self.get_custom_document_generator()
        else:
            raise NotImplementedError()

    def get_the_pile_document_generator(self) -> Generator:
        """
        This method reads input files with similar file formats with The Pile's jsonl format.
        Each line of the input file should be a json string, where the document is stored in a field named "text".
        There are no empty lines between json lines.

        Example:
        {"text": "Hello World!", "meta": {"pile_set_name": "Pile-CC"}}
        {"text": "Foo bar", "meta": {"pile_set_name": "Pile-CC"}}
        """
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)["text"]

    def get_raw_document_generator(self) -> Generator:
        """
        This method reads input files where each line is a document. The file should not be organized
        in any specific file structures such as json, jsonl, or tsv, as this may affect ngram computation.
        Any characters other than the actual text content should be removed.

        Example:
        Hello World!
        Foo bar
        This is the 3rd document.
        """
        with open(self.file_path, "r") as f:
            for line in f:
                yield line.rstrip("\n")

    def get_custom_document_generator(self) -> Generator:
        """Define your own document reading method"""
        pass
