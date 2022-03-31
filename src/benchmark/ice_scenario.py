import os
import re
import sys  # noqa
import requests  # noqa
from typing import List, Set
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, TEST_SPLIT
from enum import Enum
import pandas as pd

TAGS = False
GENDER_ANNOTATIONS = {"M": {"M", "m", "Male", "male"}, "F": {"F", "f", "Female", "female"}}


class HeaderFormat(Enum):
    NONE = 0
    HDR = 1
    XLS = 2
    MDB = 3


class ICEScenario(Scenario):
    """
    The International Corpus of English (ICE).

    Documentation: https://www.ice-corpora.uzh.ch/en.html
    """

    name = "ice"
    description = "International Corpus of English (ICE)"
    tags = ["harms", "fairness", "language_modeling"]

    def __init__(self, subset: str, split: str = "all", gender: str = None):
        ice_subsets = {"CAN", "JA", "HK", "IND", "SIN", "PHI", "USA"}
        unsupported_subsets = {"GB", "EA", "IRL", "NZ", "SL", "NG"}  # noqa
        assert subset in ice_subsets
        self.subset = subset

        assert split in {"all", "written", "spoken"}
        self.split = split

        metadata_hdr = {"IND", "USA"}
        metadata_xls = {"CAN", "HK"}

        subset_to_directory = {
            "IND": "ICE India",
            "CAN": "ICE-CAN",
            "HK": "ICE-HK",
            "JA": "ICE-JA",
            "PHI": "ICE Philippines",
            "SIN": "ICE SINGAPORE",
            "USA": "ICE-USA",
        }
        self.directory = subset_to_directory[self.subset]
        self.gender = "None"

        if gender:
            assert subset in (metadata_hdr | metadata_xls)
            self.gender = gender

            if subset in metadata_hdr:
                self.gender_mode = HeaderFormat.HDR
            elif subset in metadata_xls:
                self.gender_mode = HeaderFormat.XLS

    def preprocess_text(self, text: str, tags: bool = False):
        """
        Pre-processes each instance text according to the following procedure:
        1. String leading/trailing whitespace. If tags are kept (tags = True), return.
        2. Replace the tags in "replace" according to the provided dictionary.
        3. Remove the tags + enclosed text for the tags in "remove".
        4. Remove all other tags completely (keeping enclosed contents).
        5. Unstrip text again.

        Notes: Ambiguous choices are listed below.
        """
        # Remove <O></O>/replace with unk?
        # Replace speaker markers with text markers shortened to include speaker information?
        #   (then every line annotated for speaker)
        # Replace <mention> with ""/untag?
        # Untag <X>/Remove?
        # Keep HTML-compatible tags? (p, h, bold, it, ul)
        # Currently have no good way of handling special characters (e.g. é),
        #   which are notated differently across subsets
        untag = [
            "I",
            "ICE.+",
            "[\[\]\{\}]\d*",
            "X",
            "@",
            "quote",
            "foreign",
            "indig",
            "smallcaps",
            "roman",
            "sb",
            "ss",
            "footnote",
            ",",
            ",,",
            "p",
            "h",
            "bold",
            "it",
            "ul",
            "\+",
            "\=",
            "\?",
            "sp",
            "l",
        ]
        remove = ["O", "-", "&", ".", "fnr", "marginalia", "del", "w"]
        replace = {"mention": '"', "\*": "\*", "\*\/": "\*", "&eacute;": "é"}

        text = text.strip()

        if tags:
            return text

        for k, v in replace.items():
            text = re.sub(str(f"<\/*{k}>"), v, text)

        untag_pattern = "<\/*(" + "|".join(untag) + ")>"
        text = re.sub(untag_pattern, "", text)

        for tag in remove:
            # only works because our tags are static
            end_tag = f"</{tag}>"
            end_match = text.find(end_tag)

            while end_match >= 0:
                start_match = text[:end_match].rfind(f"<{tag}>")

                if start_match == -1:
                    start_match = end_match

                text = text[:start_match] + text[end_match + len(end_tag) :]
                end_match = text.find(end_tag)

        text = text.strip()
        return text

    def validate_file(self, filename: str) -> bool:
        if self.split == "all":
            return True

        if self.split == "written":
            return filename.startswith(("W", "w"))

        return filename.startswith(("S", "s"))

    def filter_by_metadata(self, header_dir: str) -> Set[str]:
        selected_texts = set()

        if self.gender_mode == HeaderFormat.XLS:
            if self.subset == "CAN":
                files = ["Spoken ICE-CAN metadata.xls", "Written ICE-CAN metadata.xls"]
            elif self.subset == "HK":
                files = ["HKRecords.xls"]

            for fi in files:
                dfs = pd.read_excel(os.path.join(header_dir, fi), sheet_name=[0] if self.subset == "CAN" else None)

                for df in dfs.values():
                    for i in range(len(df)):
                        if not pd.isna(df.iat[i, 0]) and any(
                            [x in GENDER_ANNOTATIONS[self.gender] for x in df.iloc[i, 1:]]
                        ):  # currently double counts texts with multiple genders
                            selected_texts.add(df.iat[i, 0] + ".txt")
        elif self.gender_mode == HeaderFormat.HDR:
            for filename in os.listdir(header_dir):
                header_path = os.path.join(self.output_path, self.directory, "Headers", filename)

                if not os.path.exists(header_path):
                    hlog(str(f"File {filename} skipped (no header found)."))
                    continue

                with open(header_path, "r") as f:
                    try:
                        text = self.preprocess_text(f.read(), TAGS)
                    except UnicodeDecodeError:
                        hlog(str(f"File {filename} skipped (unsupported header encoding)."))
                        continue

                gen_ann = re.findall("(?<=<gender>)\w+(?=<\/gender>)", text)

                if all([g in GENDER_ANNOTATIONS[self.gender] for g in gen_ann]):
                    selected_texts.add(filename[:-4] + ".txt")

        return selected_texts

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, self.directory)

        # Currently there is no infrastructure in place to unzip files with passwords
        # (the HTTP requests for the data itself also requires some authorization);
        # therefore we can only assume data is already downloaded and extracted
        ensure_file_downloaded(
            source_url=None, target_path=data_path, unpack=True,
        )

        if "Corpus" in os.listdir(data_path):
            corpus_name = "Corpus"
        elif "CORPUS" in os.listdir(data_path):
            corpus_name = "CORPUS"
        else:
            corpus_name = "corpus"

        corpus_path = os.path.join(data_path, corpus_name)
        header_dir = os.path.join(data_path, "Headers")
        selected_texts = self.filter_by_metadata(header_dir) if self.gender != "None" else os.listdir(corpus_path)
        instances = []

        for filename in selected_texts:
            if not self.validate_file(filename):
                continue

            with open(os.path.join(corpus_path, filename), "r") as f:
                try:
                    text = self.preprocess_text(f.read(), TAGS)
                except UnicodeDecodeError:
                    hlog(
                        str(f"File {filename} skipped (unsupported encoding).")
                    )  # ISO texts currently unsupported; only a small minority of texts affected
                    continue

                instances.append(Instance(input=text, references=[], split=TEST_SPLIT))

        return instances
