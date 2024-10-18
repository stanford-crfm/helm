import os
import glob
import json
from typing import Dict, List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class EchrJudgeScenario(Scenario):
    """
    Task:
    - This scenario is a binary classification task.
    - It classifies human right case description into violation or no violation.

    Dataset:
    - EN_train, EN_dev, EN_test (These data sets are downloaded).
    - These dataset are considered as TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT.
    - Each dataset is a set of JSON files containing at least TEXT and VIOLATED_ARTICLES fields.
        - TEXT fields contains sentences.
        - VIOLATED_ARTICLES contains information about
          human rights violation or no violation (in case of empty list)

    Prompt:
        ------
        Is the following case a violation of human rights?  (Instructions)

        Case: Human rights have not been violated.          (Trivial No case in instructions)
        Answer: No

        Case: Human rights have been violated.              (Trivial Yes case in instructions)
        Answer: Yes

        Case: <TEXT>                                        (In-context examples, if possible)
        Answer: <Label>                                     (Label is correct answer, Yes or No)

        ...
        Case: <TEXT>                                        (Target input text)
        Answer: <Output>                                    (Output ::= Yes | No)
        ----

    - <TEXT> parts are often too long, resulting in zero-shot predictions in many cases.
      Therefore, we have added two trivial cases to the instructions part.
    
    Reference:
        Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. 2019. 
        Neural Legal Judgment Prediction in English. 
        In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 
        pages 4317–4323, Florence, Italy. Association for Computational Linguistics.
        https://aclanthology.org/P19-1424/
        
    """

    # Names of the tasks we support
    name = "echr_judge"
    description = "Predicting Legal Decisions on Human Rights Violations in English"
    tags = ["classification", "judgement", "legal"]

    # Dataset file name
    DATASET_DOWNLOAD_URL = "https://archive.org/download/ECHR-ACL2019/ECHR_Dataset.zip"
    DATASET_NAME = "ECHR_Dataset"

    # Answer labels
    ANSWER_VIOLATION = "Yes"
    ANSWER_NO_VIOLATION = "No"

    # Prompt constants (used in adapter)
    PROMPT_INPUT = "Case"
    PROMPT_OUTPUT = "Answer"

    YES_EX = f"\n\n{PROMPT_INPUT}: Human rights have been violated.\n{PROMPT_OUTPUT}: {ANSWER_VIOLATION}"
    NO_EX = f"\n\n{PROMPT_INPUT}: Human rights have not been violated.\n{PROMPT_OUTPUT}: {ANSWER_NO_VIOLATION}"
    INST_EX = f"{NO_EX}{YES_EX}"

    PROMPT_INST = "Is the following case a violation of human rights?"  # Prompt for instructions
    PROMPT_INST_WITH_EX = f"{PROMPT_INST}{INST_EX}"  # Prompt for instructions with trivial examples

    # Methods
    def __init__(self, doc_max_length: Optional[int] = None):
        """
        Args:
            doc_max_length: Int indicating the maximum word length to filter documents.
                            Documents longer than this length are ignored.
                            NOTE: Currently uses whitespace tokenization.
        """
        super().__init__()
        self.doc_max_length = doc_max_length

    def download_data(self):
        data_dir = self.data_dir
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=os.path.join(data_dir, self.DATASET_NAME),
            unpack=True,
            unpack_type="unzip",
        )

    def check_small_word_length(self, jdata, limit_word_length):
        """
        This checks number of words in jdata < limit_word_length.
        jdata : dict of {"TEXT":str,..}, limit_word_length:int .
        """
        text = " ".join(jdata["TEXT"])
        text = text.replace("\n", " ")
        word_length = len(text.split())
        if limit_word_length is None:
            return True
        else:
            if word_length < limit_word_length:
                return True
            else:
                return False

    def check_violated_article(self, jdata):
        """
        This checks that article jdata is violation data or not,
        and returns True if jdata is violation data, otherwise False.
        jdata : dict of {"VIOLATED_ARTICLES":str list, }.
        """
        valist = jdata["VIOLATED_ARTICLES"]
        return len(valist) > 0

    def get_input_references(self, jdata, limit_word_length):
        """
        This makes dictionary for Instance data from a json data for one document.
        Instance is returned if word length of jdata's text < limit_word_length, otherwise None is returned.
        """
        text = " ".join(jdata["TEXT"])
        if self.check_small_word_length(jdata, limit_word_length):
            violated = self.check_violated_article(jdata)
            answer = self.ANSWER_VIOLATION if violated else self.ANSWER_NO_VIOLATION
            ref_correct = Reference(Output(text=answer), tags=[CORRECT_TAG])
            references = [ref_correct]
            instance = {"input": Input(text), "references": references}
            return instance
        else:
            return None

    def get_instances(self, output_path: str) -> List[Instance]:
        self.data_dir = os.path.join(output_path, "data")

        # download data
        self.download_data()

        dname_insts: Dict[str, List[Instance]] = {
            "EN_train": [],
            "EN_dev": [],
            "EN_test": [],
        }  # dataname to instance list
        dname_split = {"EN_train": TRAIN_SPLIT, "EN_dev": VALID_SPLIT, "EN_test": TEST_SPLIT}  # dataname to split

        # read json files under EN_train/, EN_dev/, and EN_test/ and convert these jsons into instances
        for dname in dname_insts:
            data_dir = self.data_dir
            target_data_dir = os.path.join(data_dir, self.DATASET_NAME, dname)
            data_dir_desc = target_data_dir + "/*.json"
            for filename in sorted(glob.glob(data_dir_desc)):
                with open(os.path.join(os.getcwd(), filename), "r") as f:
                    jdata = json.load(f)
                    # get input and references from json data
                    idata = self.get_input_references(jdata, self.doc_max_length)
                    if idata is not None:
                        # create Instance for jdata
                        instance = Instance(
                            input=idata["input"], references=idata["references"], split=dname_split[dname]
                        )
                        dname_insts[dname].append(instance)

        # all instances for json data in EN_train, EN_dev, and EN_test directories.
        all_instances: List[Instance] = []
        for dname in dname_insts:
            all_instances.extend(dname_insts[dname])
        return all_instances
