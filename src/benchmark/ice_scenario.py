import os
import re
import sys  # noqa
import requests  # noqa
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog, htrack
from .scenario import Scenario, Instance, TEST_SPLIT

TAGS = False


class ICEScenario(Scenario):
    """
    The International Corpus of English (ICE).

    Documentation: https://www.ice-corpora.uzh.ch/en.html
    """

    name = "ice"
    description = "International Corpus of English (ICE)"
    tags = ["harms", "fairness", "language_modeling"]

    def __init__(self, subset: str, gender: str = None):
        self.ice_subsets = {"CAN", "GB", "JA", "HK", "EA", "IND", "SIN", "PHI", "USA", "IRL", "NZ", "SL", "NIG"}
        assert subset in self.ice_subsets
        self.subset = subset

        subset_to_directory = {"IND": "ICE India"}
        self.directory = subset_to_directory[self.subset]

    def preprocess_text(self, text: str, tags: bool = False):
        """
        Pre-processes each instance text according to the following procedure:
        1. String leading/trailing whitespace. If tags are kept (tags = True), return.
        2. Remove text tags (those with #). If speaker annotated, leave speaker in; otherwise, remove completely.
        3. Remove the tags + enclosed text for the tags in "remove".
        4. Replace the tags in "replace" according to the provided dictionary.
        5. Remove all other tags completely (keeping enclosed contents).

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
        ]
        remove = ["O", "-", "&", ".", "fnr", "marginalia", "del", "w", "*"]
        replace = {"mention": '"', "\*": "\*", "\*\/": "\*"}

        text = text.strip()

        if tags:
            return text

        for k, v in replace.items():
            text = re.sub(str(f"<\/*{k}>"), v, text)

        untag_pattern = "<\/*(" + "|".join(untag) + ")>"
        text = re.sub(untag_pattern, "", text)
        # stack = []

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

            # for match in re.finditer(f"<\/*{tag}>", text):
            #     if match[0][1] == "/":
            #         pass
            #     else:
            #         stack.append(match[0].start())

        text = text.strip()
        return text

    @htrack(None)
    def read_in(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, self.directory)
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
        instances = []

        for filename in os.listdir(corpus_path):
            with open(os.path.join(corpus_path, filename), "r") as f:
                try:
                    text = self.preprocess_text(f.read(), TAGS)
                except UnicodeDecodeError:
                    hlog(str(f"File {filename} skipped."))
                    continue

                instances.append(Instance(input=text, references=[], split=TEST_SPLIT))

        return instances
