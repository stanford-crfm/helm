import json

from typing import Iterator


def get_document_iterator(file_path: str, file_format: str) -> Iterator[str]:
    if file_format == "the_pile":
        return get_the_pile_document_iterator(file_path)
    elif file_format == "raw":
        return get_raw_document_iterator(file_path)
    elif file_format == "custom":
        return get_custom_document_iterator(file_path)
    else:
        raise NotImplementedError()


def get_the_pile_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with The Pile's jsonl format.
    Each line of the input file should be a json string, where the document is stored in a field named "text".
    There are no empty lines between json lines.

    Example:
    {"text": "Hello World!", "meta": {"pile_set_name": "Pile-CC"}}
    {"text": "Foo bar", "meta": {"pile_set_name": "Pile-CC"}}
    """
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)["text"]


def get_raw_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files where each line is a document. The file should not be organized
    in any specific file structures such as json, jsonl, or tsv, as this may affect ngram computation.
    Any characters other than the actual text content should be removed.

    Example:
    Hello World!
    Foo bar
    This is the 3rd document.
    """
    with open(file_path, "r") as f:
        for line in f:
            yield line.rstrip("\n")


def get_custom_document_iterator(file_path: str) -> Iterator[str]:
    """Define your own document reading method"""
    pass
