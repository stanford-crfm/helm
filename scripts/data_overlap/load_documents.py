import json

from typing import Iterator


def get_document_iterator(file_path: str, file_format: str) -> Iterator[str]:
    if file_format in ["the_pile", "oig", "c4"]:
        return get_the_pile_document_iterator(file_path)
    elif file_format == "flan":
        return get_flan_document_iterator(file_path)
    elif file_format in ["xp3", "p3"]:
        return get_xp3_document_iterator(file_path)
    elif file_format == "root":
        return get_root_document_iterator(file_path)
    elif file_format == "natural":
        return get_natural_document_iterator(file_path)
    elif file_format == "hh":
        return get_root_document_iterator(file_path)
    elif file_format == "raw":
        return get_raw_document_iterator(file_path)
    elif file_format == "custom":
        return get_custom_document_iterator(file_path)
    else:
        raise NotImplementedError()


def get_hh_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with Anthropic/hh-rlhh jsonl format.

    example:
    {'chosen': '\n\nHuman: What kind of noises did ...'
    'rejected': '\n\nHuman: What kind of noises...'}

    """
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)["chosen"]
            yield json.loads(line)["rejected"]


def get_natural_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with natural instructions json format.

    """
    with open(file_path) as f:
        json_dict = json.load(f)
        for example in json_dict["Positive Examples"]:
            yield example["input"]
            yield example["output"]
            yield example["explanation"]
        for example in json_dict["Negative Examples"]:
            yield example["input"]
            yield example["output"]
            yield example["explanation"]


def get_root_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with root's parquet format.

    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches():
        df = batch.to_pandas()
        for row in df.iterrows():
            yield row[1].tolist()[0]


def get_xp3_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with xp3's jsonl format.

    Example:
    {"inputs":"( 1 ) Benzoyl peroxide and ..."}

    """
    with open(file_path, "r") as f:
        for line in f:
            json_dict = json.loads(line)
            yield json_dict["inputs"]
            yield json_dict["targets"]


def get_flan_document_iterator(file_path: str) -> Iterator[str]:
    """
    This method reads input files with similar file formats with FLAN's jsonl format.

    Example:
    {'id': 'task587-712049dced9641209d4f3ba815616a8f',
    'input': "I was looking for ..."
    'output': 'False'}

    """
    with open(file_path, "r") as f:
        for line in f:
            json_dict = json.loads(line)
            for sample in json_dict["sample"]:
                yield sample["input"]
                yield sample["output"]


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
    raise NotImplementedError()
