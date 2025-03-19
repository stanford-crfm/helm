import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional

from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)
from helm.common.general import ensure_file_downloaded, ensure_directory_exists


def _strip_string(str: str) -> Any:
    # from https://stackoverflow.com/a/4703508
    numeric_const_pattern = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
    match = re.search(numeric_const_pattern, str)
    if match:
        try:
            return float(str[match.start() : match.end()])
        except Exception:
            return None
    return None


def float_equiv(str1: Optional[str], str2: Optional[str], eps: float = 1e-6) -> float:
    """
    extract the first numbers in the two strings and compare them
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        print(f"{str1}: ({ss1}) == {str2}: ({ss2})? {float(abs(ss1 - ss2) < eps)}")

        if ss1 is None or ss2 is None:
            return 0.0
        return float(abs(ss1 - ss2) < eps)
    except Exception:
        return float(str1 == str2)


class ConvFinQACalcScenario(Scenario):
    """"A mathematical calculation benchmark based on ConvFinQA.

    Data source:
    https://github.com/czyssrs/ConvFinQA

    Reference:
    Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022.
    ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering.
    In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing,
    pages 6279–6292, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
    https://aclanthology.org/2022.emnlp-main.421
    """  # noqa: E501

    name = "conv_fin_qa_calc"
    description = "A mathematical calculation benchmark based on ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering [(Chen ey al., 2022)](https://arxiv.org/pdf/2210.03849.pdf)."  # noqa: E501
    tags = ["question_answering", "finance"]

    """ Class variables """
    # Dataset file name
    DATASET_DOWNLOAD_URL: str = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

    _SPLIT_TO_JSON_FILE_NAME: Dict[str, str] = {
        TRAIN_SPLIT: "train_turn.json",
        VALID_SPLIT: "dev_turn.json"
    }

    def get_table_text(self, table: List[List[str]]) -> str:
        """table in the format of List of columns"""
        return "~".join(["|".join(col) for col in table])

    def make_pseudo_markdown_table(self, array, line_sep="\n"):
        markdown = str("|")

        for e in array[0]:
            to_add = " " + str(e) + str(" |")
            markdown += to_add
            markdown += line_sep

        for entry in array[1:]:
            markdown += str("| ")
            for e in entry:
                to_add = str(e) + str(" | ")
                markdown += to_add
                markdown += line_sep

        return markdown

    def get_instance_dict(self, dic, split: str, sep: str = "\n") -> Instance:
        linearized_table = self.make_pseudo_markdown_table(dic["table"], line_sep=sep)

        if "gold_ind" in dic["annotation"]:
            facts = dic["annotation"]["gold_ind"]
        elif "gold_inds" in dic["annotation"]:
            facts = dic["annotation"]["gold_inds"]
        else:
            facts = {}

        text = ""
        for fact_type, fact in facts.items():
            if "text" in fact_type:
                text += fact
        context = ""
        for ind, q in enumerate(dic["annotation"]["cur_dial"]):
            if ind < len(dic["annotation"]["cur_dial"]) - 1:
                context += q + " The answer is " + str(dic["annotation"]["exe_ans_list"][ind]) + " " + sep
            else:
                context += q + " The answer is "
        doc = f"Table: {sep}{linearized_table}{sep}Text: {text}{sep}Questions: "
        answer = str(dic["annotation"]["exe_ans"])
        return Instance(
            input=Input(text="".join(doc) + " " + context),
            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
            split=split,
        )

    def load_dataset(self, output_path: str) -> Tuple[List[Instance], List[Instance]]:
        """Loads the dataset downloaded in download_dataset()."""
        folder_path = os.path.join(output_path, "data")
        train_data = []
        dev_data = []

        with open(os.path.join(folder_path, "train_turn.json"), encoding="utf-8") as f:
            train_raw_data = json.load(f)

        for problem in train_raw_data:
            train_data.append(self.get_instance_dict(problem, TRAIN_SPLIT))

        with open(os.path.join(folder_path, "dev_turn.json"), encoding="utf-8") as f:
            dev_raw_data = json.load(f)

        for problem in dev_raw_data:
            dev_data.append(self.get_instance_dict(problem, VALID_SPLIT))

        return train_data, dev_data

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=os.path.join(output_path, "data"),
            unpack=True,
            unpack_type="unzip",
        )

        train_data, dev_data = self.load_dataset(output_path)
        print(len(train_data))
        print(len(dev_data))
        return train_data + dev_data
