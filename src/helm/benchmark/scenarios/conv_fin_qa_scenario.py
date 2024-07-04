import json
import os
from typing import Dict, List, Tuple, Any, Optional
import re

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


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


class ConvFinQAScenario(Scenario):
    """
    TODO: edit the discription

    FiQA-2018 Financial Opinion Mining and Question Answering Task 2: https://sites.google.com/view/fiqa/home

    The data description from the original site:
    " Given a corpus of structured and unstructured text documents from different financial data
    sources in English (microblogs, reports, news) build a Question Answering system that answers
    natural language questions. For this challenge, part of the questions will be opinionated,
    targeting mined opinions and their respective entities, aspects, sentiment polarity and
    opinion holder. "

    The data is organized in the format of question pool and answer pool, and the task is to rank
    relevant documents from the reference knowledge base with regard to a natural language question.
    In order to make the problem more tractable, we make it a multi choice question with the correct
    answer included in the options mixed with other random selected answers.


    {
      "question": "What is considered a business expense on a business trip?",
      "answer": "The IRS Guidance pertaining to the subject.  In general the best I can say is your business expense may be deductible.  But it depends on the circumstances and what it is you want to deduct. Travel Taxpayers who travel away from home on business may deduct related   expenses, including the cost of reaching their destination, the cost   of lodging and meals and other ordinary and necessary expenses.   Taxpayers are considered “traveling away from home” if their duties   require them to be away from home substantially longer than an   ordinary day’s work and they need to sleep or rest to meet the demands   of their work. The actual cost of meals and incidental expenses may be   deducted or the taxpayer may use a standard meal allowance and reduced   record keeping requirements. Regardless of the method used, meal   deductions are generally limited to 50 percent as stated earlier.    Only actual costs for lodging may be claimed as an expense and   receipts must be kept for documentation. Expenses must be reasonable   and appropriate; deductions for extravagant expenses are not   allowable. More information is available in Publication 463, Travel,   Entertainment, Gift, and Car Expenses. Entertainment Expenses for entertaining clients, customers or employees may be   deducted if they are both ordinary and necessary and meet one of the   following tests: Directly-related test: The main purpose of the entertainment activity is the conduct of business, business was actually conducted   during the activity and the taxpayer had more than a general   expectation of getting income or some other specific business benefit   at some future time.   Associated test: The entertainment was associated with the active conduct of the taxpayer’s trade or business and occurred directly   before or after a substantial business discussion. Publication 463 provides more extensive explanation of these tests as   well as other limitations and requirements for deducting entertainment   expenses. Gifts Taxpayers may deduct some or all of the cost of gifts given in the   course of their trade or business. In general, the deduction is   limited to $25 for gifts given directly or indirectly to any one   person during the tax year. More discussion of the rules and   limitations can be found in Publication 463. If your LLC reimburses you for expenses outside of this guidance it should be treated as Income for tax purposes. Edit for Meal Expenses: Amount of standard meal allowance.   The standard meal allowance is   the federal M&IE rate. For travel in 2010, the rate for most small   localities in the United States is $46 a day. Source IRS P463 Alternately you could reimburse at a per diem rate",
      "meta_info": "step2&3",
      "answer_idx": "A"
    }
    """  # noqa

    """ Information on this class"""
    name = "conv_fin_qa"
    description = "Conversitional Finance QA"
    tags = ["question_answering", "finance"]

    """ Class variables """
    # Dataset file name
    DATASET_DOWNLOAD_URL: str = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    DATASET_FILE_NAME = "ConvFinQA"

    def __init__(self):
        super().__init__()

    def download_dataset(self, output_path: str):
        """Downloads the con_fin_qa dataset."""

        # Download the raw data
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=os.path.join(data_dir, self.DATASET_FILE_NAME),
            unpack=True,
            unpack_type="unzip",
        )

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

    def get_instance_dict(self, dic, sep: str = "\n") -> Dict[str, Any]:
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
        return {
            "input": PassageQuestionInput(passage="".join(doc), question=context, separator=" "),
            "references": [Reference(Output(text=answer), tags=[CORRECT_TAG])],
        }

    def load_dataset(self, output_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Loads the dataset downloaded in download_dataset()."""
        folder_path = os.path.join(output_path, "data", self.DATASET_FILE_NAME)
        train_data = []
        dev_data = []

        with open(os.path.join(folder_path, "train_turn.json"), encoding="utf-8") as f:
            train_raw_data = json.load(f)

        for problem in train_raw_data:
            train_data.append(self.get_instance_dict(problem))

        with open(os.path.join(folder_path, "dev_turn.json"), encoding="utf-8") as f:
            dev_raw_data = json.load(f)

        for problem in dev_raw_data:
            dev_data.append(self.get_instance_dict(problem))

        return train_data, dev_data

    def get_instances(self, output_path: str) -> List[Instance]:
        """Returns the instances for this scenario."""
        # Body of the function
        self.download_dataset(output_path)
        train_data, dev_data = self.load_dataset(output_path)
        train_k = 5
        train_instances = [
            Instance(input=d["input"], references=d["references"], split=TRAIN_SPLIT) for d in train_data[:train_k]
        ]
        valid_instances = [
            Instance(input=d["input"], references=d["references"], split=VALID_SPLIT) for d in dev_data[:1000]
        ]
        print("length of validate:", len(valid_instances))
        return train_instances + valid_instances
