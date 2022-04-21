"""Assortment of useful utility functions. """

import os
import ujson as json
from typing import Iterator, Dict, List
import pandas as pd
from pathlib import Path


def jsonl_generator(fname: str) -> Iterator[Dict[str, str]]:
    """Returns an iterator over a jsonl file."""
    for line in open(fname, "r"):
        line = line.strip()
        if len(line) < 3:
            d = {}
        elif line[len(line) - 1] == ",":
            d = json.loads(line[: len(line) - 1])
        else:
            d = json.loads(line)
        yield d


def batch_line_generator(fname, batch_size):
    """ Returns generator for jsonl file with batched lines """
    res = []
    batch_id = 0
    for line in open(fname, "r"):
        line = line.strip()
        if len(line) < 3:
            d = ""
        elif line[len(line) - 1] == ",":
            d = line[: len(line) - 1]
        else:
            d = line
        res.append(d)
        if len(res) >= batch_size:
            yield batch_id, res
            batch_id += 1
            res = []
    yield batch_id, res


def append_to_jsonl_file(data, file):
    """ Appends json dictionary as new line to file """
    with open(file, "a+") as out_file:
        for x in data:
            out_file.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_batch_files(fdir: Path) -> List[str]:
    """For each file in fdir, returns full filepath.

    Args:
        fdir (str): path to directory

    Returns:
        List[str]: filepaths for files in fdir
    """
    return list(fdir.iterdir())


def create_dir(out_dir):
    """ Creates new directory if it doesn't already exist """
    if not os.path.exists(out_dir):
        print(f"Creating {out_dir}")
        os.makedirs(out_dir)


def load_seed_relations(fdir: Path) -> pd.DataFrame:
    """
    Returns a dataframe containing seed relations and associated information.

    Args:
        fdir (str): path to folder containined seed relations TSV files

    Returns:
        pd.DataFrame: dataframe for data in tsv files.
    """
    filepaths = fdir.glob("*.tsv")
    df = pd.DataFrame()
    for filepath in filepaths:
        df = df.append(pd.read_csv(filepath, sep="\t"))
    return df
