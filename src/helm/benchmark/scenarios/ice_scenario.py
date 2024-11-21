import os
import re
from typing import List, Union
from enum import Enum
import pandas as pd

from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.ice_scenario_pinned_file_order import listdir_with_pinned_file_order
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input

try:
    # pd.read_excel() uses xlrd
    import xlrd  # noqa
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["scenarios"])


class ICESubset(Enum):
    CANADA = "can"
    JAMAICA = "ja"
    HONG_KONG = "hk"
    INDIA = "ind"
    SINGAPORE = "sin"
    PHILIPPINES = "phi"
    USA = "usa"
    EAST_AFRICA = "ea"
    IRELAND = "irl"


class TextCategory(Enum):
    ALL = "all"
    S_ALL = "S"
    W_ALL = "W"
    S_DIALOGUE = "S1"
    S_MONOLOGUE = "S2"
    W_PRIVATE = "W1"
    W_PUBLIC = "W2"


class BinaryGender(Enum):
    MALE = "male"
    FEMALE = "female"


class HeaderFormat(Enum):
    NONE = 0
    HDR = 1
    XLS = 2
    MDB = 3


KEEP_TAGS = False
GENDER_ANNOTATIONS = {
    BinaryGender.MALE: {"M", "m", "Male", "male"},
    BinaryGender.FEMALE: {"F", "f", "Female", "female"},
}
UNSUPPORTED_SUBSETS = {"GB", "NIG", "NZ", "SL"}
METADATA_FORMAT = {
    ICESubset.CANADA: HeaderFormat.XLS,
    ICESubset.HONG_KONG: HeaderFormat.XLS,
    ICESubset.INDIA: HeaderFormat.HDR,
    ICESubset.USA: HeaderFormat.HDR,
}
SUBSET_TO_DIRECTORY = {
    ICESubset.INDIA: "ICE India",
    ICESubset.CANADA: "ICE-CAN",
    ICESubset.HONG_KONG: "ICE-HK",
    ICESubset.JAMAICA: "ICE-JA",
    ICESubset.PHILIPPINES: "ICE Philippines",
    ICESubset.SINGAPORE: "ICE SINGAPORE",
    ICESubset.USA: "ICE-USA",
    ICESubset.IRELAND: "ICE-IRL/ICE-Ireland version 1.2#6DAE.2/ICE-Ireland txt",
    ICESubset.EAST_AFRICA: "ICE-EA/corpus/retagged for wsmith",
}
EA_FILENAME_TO_CATEGORY = {
    "brdisc": TextCategory.S_DIALOGUE,
    "brint": TextCategory.S_DIALOGUE,
    "brnews": TextCategory.S_MONOLOGUE,
    "brtalk": TextCategory.S_MONOLOGUE,
    "buslet": TextCategory.W_PRIVATE,
    "clless": TextCategory.S_DIALOGUE,
    "column": TextCategory.W_PUBLIC,
    "conv": TextCategory.S_DIALOGUE,
    "conv1": TextCategory.S_DIALOGUE,
    "conv2": TextCategory.S_DIALOGUE,
    "crea": TextCategory.W_PUBLIC,
    "creat1": TextCategory.W_PUBLIC,
    "creat2": TextCategory.W_PUBLIC,
    "crossx": TextCategory.S_DIALOGUE,
    "editor": TextCategory.W_PUBLIC,
    "essays": TextCategory.W_PRIVATE,
    "exam": TextCategory.W_PRIVATE,
    "feat": TextCategory.W_PUBLIC,
    "instruct": TextCategory.W_PUBLIC,
    "judge": TextCategory.W_PRIVATE,
    "ldhum": TextCategory.W_PUBLIC,
    "ldnats": TextCategory.W_PUBLIC,
    "ldsoc": TextCategory.W_PUBLIC,
    "ldtech": TextCategory.W_PUBLIC,
    "parlia": TextCategory.S_DIALOGUE,
    "ppgen": TextCategory.W_PUBLIC,
    "pphum": TextCategory.W_PUBLIC,
    "ppnats": TextCategory.W_PUBLIC,
    "ppsoc": TextCategory.W_PUBLIC,
    "pptech": TextCategory.W_PUBLIC,
    "schbr": TextCategory.S_MONOLOGUE,
    "soclet": TextCategory.W_PRIVATE,
    "splect": TextCategory.S_MONOLOGUE,
    "splash": TextCategory.W_PUBLIC,
}


class ICEScenario(Scenario):
    """
    The International Corpus of English (ICE).

    NOTE: This text cannot be downloaded automatically.
    You must extract each subset zip file into args.output_path + '/scenarios/ice',
    which is by default '/benchmark_output/scenarios/ice',
    where args.output_path is parsed from the command line argument.
    See helm.benchmark.runner for more details about args.output_path.

    The archives should extract into folders named according to the dictionary SUBSET_TO_DIRECTORY
    below.

    The ICE corpus gathers written and spoken texts from variants of English across 13
    regional subsets:
    Canada, East Africa (Kenya & Tanzania), Great Britain, Hong Kong, India, Ireland,
    Jamaica, Nigeria, New Zealand, the Philippines, Singapore, Sri Lanka, and the United States.
    We evaluate on per-text perplexity (by default, all texts from all regions, but
    can be filtered using scenario parameters).

    Initially, we are only able to evaluate the Canada (can), Hong Kong (hk), India (ind), Jamaica (ja),
    Philippines (phi), Singapore (sin) and United States (usa) subsets, as these are the
    only subsets which standardize the organization of their data/metadata. Evaluation
    can be restricted to one of these subsets by passing the corresponding code (parenthesized above)
    into the subset parameter.

    Spoken texts are transcripts of conversations, speeches or radio/television programs,
    while written texts range over essays, emails, news reports and other professional
    written material. The corpus is marked up with XML-style annotations which we have
    chosen to eliminate (save for the speaker annotations in the spoken texts).

    Here is a spoken text example (from ICE India):

    ```
    <|endoftext|><$A>

    He says one minute


    About that uh mm letter sir


    About uh that letter


    Board of studies letter

    <$B>

    I gave it you no
    ...
    ```

    Here is a written text example (from ICE-USA):

    ```
    <|endoftext|>The U.S. Mint:



      United States coins are made at four Mint facilities:

    Philadelphia, Denver, San Francisco, and West Point, NY.
     One easy way to start your collection is with the circulating coins

    you use daily - pennies, nickels, dimes, quarters and dollars.
     In addition, the U.S. Mint also issues annual proof and uncirculated
    ...
    ```

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

    def __init__(self, subset: Union[str, None] = None, gender: Union[str, None] = None, category="all"):
        super().__init__()
        if subset:
            self.subset = [ICESubset(subset)]
        else:
            self.subset = list(ICESubset)

        self.gender = None

        if gender:
            if subset:
                assert (
                    ICESubset(subset) in METADATA_FORMAT.keys()
                ), f"Subset {subset} cannot be filtered through metadata."
            else:
                self.subset = list(METADATA_FORMAT.keys())

            self.gender = BinaryGender(gender)

        self.category = TextCategory(category)

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
            "[\\[\\]\\{\\}]\\d*",
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
            "p",
            "h",
            "bold",
            "it",
            "ul",
            "\\+",
            "\\=",
            "\\?",
            "sp",
            "}",
            "ea",
            "slang",
        ]
        remove = ["O", "-", "&", ".", "fnr", "marginalia", "del", "w"]
        delete = [
            "[WS]\\d[A-F]-\\d{3} *",
            " <l> ",
            "<#> ",
            "BBB\\S+ \\S+RRR *",
            "BBB\\S+ *",
            "RRR\\S+ *",
            "UUU",
            "III\\S+ *",
        ]
        replace = {
            "<\\/*mention>": '"',
            "</\\*\\*>": "*",
            "</\\*\\*/>": "*",
            "&eacute;": "é",
            " <,+>": ",",
            "&ersand;": "&",
            "&obrack;": "(",
            "&cbrack;": ")",
            "&percent;": "%",
            "&hash;": "#",
            "&dash": "-",
            "&ldquo;": '"',
            "&atsign;": "@",
            "\\n+[\\n ]+": "\n\n",
            "\\S+><\\+_(\\S+)>": "\\1",
        }

        text = text.strip()

        if keep_tags:
            return text

        for i in delete:
            text = re.sub(i, "", text)

        for k, v in replace.items():
            text = re.sub(k, v, text)

        untag_pattern = "<\\/*(" + "|".join(untag) + ")>"
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

    def validate_category(self, subset: ICESubset, filename: str) -> bool:
        """
        Returns true if the text specified by @param filename belongs to the
        category specified in the scenario parameters. In standard ICE formatting,
        text names begin with W if written and S if spoken.
        """
        if self.category == TextCategory.ALL:
            return True

        if subset == ICESubset.EAST_AFRICA:
            name = filename.replace("-", "")[:-5].lower()
            category = EA_FILENAME_TO_CATEGORY[name]

            if self.category == TextCategory.S_ALL:
                return category in [TextCategory.S_DIALOGUE, TextCategory.S_MONOLOGUE]
            elif self.category == TextCategory.W_ALL:
                return category in [TextCategory.W_PRIVATE, TextCategory.W_PUBLIC]
            else:
                return self.category == category

        return filename.upper().startswith(self.category.value)

    def filter_by_metadata(self, subset: ICESubset, header_dir: str, all_corpus_filenames: List[str]) -> List[str]:
        """
        Reads through metadata in a folder specified by @param header_dir
        and returns a set of corresponding text filenames (in the Corpus folder)
        which satisfy the constraints specified by the scenario parameters.
        E.g. if gender == "M", the output set will only contain filenames for
        texts with only male authors.
        """
        assert self.gender, "Tried to filter without gender specified!"

        selected_texts = set()

        if METADATA_FORMAT[subset] == HeaderFormat.XLS:
            if subset == ICESubset.CANADA:
                files = [("Spoken ICE-CAN metadata.xls", [12]), ("Written ICE-CAN metadata.xls", [12, 19])]
            elif subset == ICESubset.HONG_KONG:
                files = [("HKRecords.xls", [16])]
            else:
                return []

            for fi, columns in files:
                dfs = pd.read_excel(
                    os.path.join(header_dir, fi), sheet_name=[0] if subset == ICESubset.CANADA else None
                )

                for df in dfs.values():
                    for i in range(len(df)):
                        if not pd.isna(df.iat[i, 0]) and any(
                            [df.iat[i, c] in GENDER_ANNOTATIONS[self.gender] for c in columns]
                        ):
                            selected_texts.add(df.iat[i, 0])
        elif METADATA_FORMAT[subset] == HeaderFormat.HDR:
            for filename in os.listdir(header_dir):
                if not filename.endswith("hdr"):
                    continue

                header_path = os.path.join(header_dir, filename)

                if not os.path.exists(header_path):
                    continue

                try:
                    with open(header_path, "r") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(header_path, "r", encoding="iso-8859-1") as f:
                        text = f.read()

                gen_ann = re.findall("(?<=<gender>)\\w+(?=<\\/gender>)", text)

                if len(gen_ann) and all([g in GENDER_ANNOTATIONS[self.gender] for g in gen_ann]):
                    selected_texts.add(filename[:-4])

        regexes = [re.compile(code, re.IGNORECASE) for code in selected_texts]
        corpus_filenames = list(filter(lambda x: any([regex.match(x) for regex in regexes]), all_corpus_filenames))
        return sorted(corpus_filenames)

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []

        for subset in self.subset:
            # Currently there is no infrastructure in place to unzip files with passwords
            # (the HTTP requests for the data itself also requires some authorization);
            # therefore we can only assume data is already downloaded and extracted
            data_path: str = os.path.join("restricted", self.name, SUBSET_TO_DIRECTORY[subset])
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
            elif subset == ICESubset.IRELAND:
                corpus_name = "ICE combined running txt"
            elif subset == ICESubset.EAST_AFRICA:
                corpus_name = "all east africa"
            else:
                corpus_name = "corpus"

            corpus_path = os.path.join(data_path, corpus_name)

            can_filter = subset in list(METADATA_FORMAT.keys()) and self.gender
            selected_texts = (
                self.filter_by_metadata(
                    subset,
                    os.path.join(data_path, "Headers"),
                    listdir_with_pinned_file_order(output_path, corpus_path),
                )
                if can_filter
                else listdir_with_pinned_file_order(output_path, corpus_path)
            )

            for filename in selected_texts:
                if not self.validate_category(subset, filename):
                    continue

                try:
                    with open(os.path.join(corpus_path, filename), "r") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(os.path.join(corpus_path, filename), "r", encoding="iso-8859-1") as f:
                        text = f.read()

                preprocessed_texts: List[str]
                if subset == ICESubset.EAST_AFRICA:
                    texts: List[str] = re.split("[WS]\\d[A-F]\\d{3}[A-Z]*[KT]\n", text)
                    preprocessed_texts = [self.preprocess_text(t, KEEP_TAGS) for t in texts if len(t) > 0]
                else:
                    preprocessed_texts = [self.preprocess_text(text, KEEP_TAGS)]

                for t in preprocessed_texts:
                    instances.append(Instance(Input(text=t), references=[], split=TEST_SPLIT))

        return instances
