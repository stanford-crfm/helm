import os
import re
import sys  # noqa
import requests  # noqa
from typing import List, Set
from enum import Enum
import pandas as pd

from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, TEST_SPLIT


class HeaderFormat(Enum):
    NONE = 0
    HDR = 1
    XLS = 2
    MDB = 3


KEEP_TAGS = False
GENDER_ANNOTATIONS = {"M": {"M", "m", "Male", "male"}, "F": {"F", "f", "Female", "female"}}
ICE_SUBSETS = {"CAN", "JA", "HK", "IND", "SIN", "PHI", "USA"}
UNSUPPORTED_SUBSETS = {"GB", "EA", "IRL", "NZ", "SL", "NG"}  # noqa
METADATA_FORMAT = {"CAN": HeaderFormat.XLS, "HK": HeaderFormat.XLS, "IND": HeaderFormat.HDR, "USA": HeaderFormat.HDR}
SUBSET_TO_DIRECTORY = {
    "IND": "ICE India",
    "CAN": "ICE-CAN",
    "HK": "ICE-HK",
    "JA": "ICE-JA",
    "PHI": "ICE Philippines",
    "SIN": "ICE SINGAPORE",
    "USA": "ICE-USA",
}


class ICEScenario(Scenario):
    """
    The International Corpus of English (ICE).

    NOTE: This text cannot be downloaded
    automatically. You must extract each subset zip file into /benchmark_output/scenarios/ice.
    The archives should extract into folders named according to the dictionary SUBSET_TO_DIRECTORY
    below.

    The ICE corpus gathers written and spoken texts from variants of English across 13
    regional subsets:
    Canada, East Africa (Kenya & Tanzania), Great Britain, Hong Kong, India, Ireland,
    Jamaica, Nigeria, New Zealand, the Philippines, Singapore, Sri Lanka, and the United States.
    We evaluate on per-text perplexity (by default, all texts from all regions, but
    can be filtered using scenario parameters).

    Initially, we are only able to evaluate the Canada (CAN), Hong Kong (HK), India (IND), Jamaica (JA),
    Philippines (PHI), Singapore (SIN) and United States (USA) subsets, as these are the
    only subsets which standardize the organization of their data/metadata. Evaluation
    can be restricted to one of these subsets by passing the corresponding code (parenthesized above)
    into the subset parameter.

    Spoken texts are transcripts of conversations, speeches or radio/television programs,
    while written texts range over essays, emails, news reports and other professional
    written material. The corpus is marked up with XML-style annotations which we have
    chosen to eliminate (save for the speaker annotations in the spoken texts).

    Here is a spoken text example (from ICE India):
    <|endoftext|><$A>

    He says one minute


    About that uh mm letter sir


    About uh that letter


    Board of studies letter

    <$B>

    I gave it you no
    ...

    Here is a written text example (from ICE-USA):
    <|endoftext|>The U.S. Mint:



      United States coins are made at four Mint facilities:

    Philadelphia, Denver, San Francisco, and West Point, NY.
     One easy way to start your collection is with the circulating coins

    you use daily - pennies, nickels, dimes, quarters and dollars.
     In addition, the U.S. Mint also issues annual proof and uncirculated
    ...

    Each subset contains exactly 500 texts and maintains a standardized distribution across categories.
    One notable exception to this distribution is the USA subset, for which the spoken texts
    are not present. Evaluation can be restricted to written or spoken texts by passing
    "written" or "spoken" respectively to the split parameter.

    Some subsets record metadata of the author(s)/speaker(s) of each text. Currently,
    CAN, HK, IND, USA support filtering texts by gender (gender=M for male, F for female).
    Where there are multiple authors/speakers, a text is only included if all the authors/speakers
    are identified with a single gender. We plan to add support for metadata filtering in PHI,
    as well as filtering by speaker age groups.

    Further documentation is provided at https://www.ice-corpora.uzh.ch/en.html
    """

    name = "ice"
    description = "International Corpus of English (ICE)"
    tags = ["language_modeling", "harms", "fairness"]

    def __init__(self, subset: str = None, split: str = "all", gender: str = None):
        if subset:
            assert subset in ICE_SUBSETS
            self.subset = {subset}
        else:
            self.subset = ICE_SUBSETS

        assert split in {"all", "written", "spoken"}
        self.split = split
        self.gender = "None"

        if gender:
            if subset:
                assert subset in METADATA_FORMAT.keys()
            else:
                self.subset = set(METADATA_FORMAT.keys())

            self.gender = gender

    def preprocess_text(self, text: str, keep_tags: bool = False) -> str:
        """
        Reads in the fulltext string of a corpus text and returns a preprocessed
        version (also string) according to the following procedure:
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
        ]
        remove = ["O", "-", "&", ".", "fnr", "marginalia", "del", "w"]
        replace = {"<\/*mention>": '"', "<\/*\*>": "<\/*\*>", "<\/*\*\/>": "\*", "&eacute;": "é", " <l> ": ""}

        text = text.strip()

        if keep_tags:
            return text

        for k, v in replace.items():
            text = re.sub(k, v, text)

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

    def validate_split(self, filename: str) -> bool:
        """
        Returns true if the text specified by @param filename belongs to the
        split specified in the scenario parameters. In standard ICE formatting,
        text names begin with W if written and S if spoken.
        """
        if self.split == "all":
            return True

        if self.split == "written":
            return filename.startswith(("W", "w"))

        return filename.startswith(("S", "s"))

    def filter_by_metadata(self, subset: str, header_dir: str) -> Set[str]:
        """
        Reads through metadata in a folder specified by @param header_dir
        and returns a set of corresponding text filenames (in the Corpus folder)
        which satisfy the constraints specified by the scenario parameters.
        E.g. if gender == "M", the output set will only contain filenames for
        texts with only male authors.
        """
        selected_texts = set()

        if METADATA_FORMAT[subset] == HeaderFormat.XLS:
            if subset == "CAN":
                files = ["Spoken ICE-CAN metadata.xls", "Written ICE-CAN metadata.xls"]
            elif subset == "HK":
                files = ["HKRecords.xls"]

            for fi in files:
                dfs = pd.read_excel(os.path.join(header_dir, fi), sheet_name=[0] if subset == "CAN" else None)

                for df in dfs.values():
                    for i in range(len(df)):
                        if not pd.isna(df.iat[i, 0]) and any(
                            [x in GENDER_ANNOTATIONS[self.gender] for x in df.iloc[i, 1:]]
                        ):  # currently double counts texts with multiple genders
                            selected_texts.add(df.iat[i, 0] + ".txt")
        elif METADATA_FORMAT[subset] == HeaderFormat.HDR:
            for filename in os.listdir(header_dir):
                header_path = os.path.join(self.output_path, SUBSET_TO_DIRECTORY[subset], "Headers", filename)

                if not os.path.exists(header_path):
                    hlog(str(f"File {filename} skipped (no header found)."))
                    continue

                with open(header_path, "r") as f:
                    try:
                        text = f.read()
                    except UnicodeDecodeError:
                        hlog(str(f"File {filename} skipped (unsupported header encoding)."))
                        continue

                gen_ann = re.findall("(?<=<gender>)\w+(?=<\/gender>)", text)

                if all([g in GENDER_ANNOTATIONS[self.gender] for g in gen_ann]):
                    selected_texts.add(filename[:-4] + ".txt")

        return selected_texts

    def get_instances(self) -> List[Instance]:
        instances = []

        for subset in self.subset:
            data_path = os.path.join(self.output_path, SUBSET_TO_DIRECTORY[subset])

            # Currently there is no infrastructure in place to unzip files with passwords
            # (the HTTP requests for the data itself also requires some authorization);
            # therefore we can only assume data is already downloaded and extracted
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Data does not exist at the required location. \
                        Please extract the relevant subset into {data_path}."
                )

            corpus_contents = os.listdir(data_path)

            if "Corpus" in corpus_contents:
                corpus_name = "Corpus"
            elif "CORPUS" in corpus_contents:
                corpus_name = "CORPUS"
            else:
                corpus_name = "corpus"

            corpus_path = os.path.join(data_path, corpus_name)
            header_dir = os.path.join(data_path, "Headers")
            selected_texts = (
                self.filter_by_metadata(subset, header_dir) if self.gender != "None" else os.listdir(corpus_path)
            )

            for filename in selected_texts:
                if not self.validate_split(filename):
                    continue

                try:
                    with open(os.path.join(corpus_path, filename), "r") as f:
                        text = self.preprocess_text(f.read(), KEEP_TAGS)
                except UnicodeDecodeError:
                    with open(os.path.join(corpus_path, filename), "r", encoding="iso-8859-1") as f:
                        text = self.preprocess_text(f.read(), KEEP_TAGS)

                instances.append(Instance(input=text, references=[], split=TEST_SPLIT))

        return instances
