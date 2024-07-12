import os
import random
from typing import List, Tuple, Dict, Any
import json
import itertools
import logging
import re

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

# TAG_DICT = {
#     "kpi": "Key Performance Indicators expressible in numerical and monetary value, e.g. revenue or net sales.",
#     "cy": "Current Year monetary value of a KPI .",
#     "py": "Prior Year monetary value of a KPI.",
#     "py1": "2 Year Past Value of a KPI",
#     "increase": "Increase of a KPI from the previous year to the current year.",
#     "increase-py": "Analogous to increase, but from py1 to py.",
#     "decrease": "Decrease of a KPI from the previous year to the current year.",
#     "decrease-py": "Analogous to decrease, but from py1 to py.",
#     "thereof": "Represents a subordinate KPI, i.e. if a KPI is part of another, broader KPI.",
#     "attr": "Attribute that further describes a KPI.",
#     "kpi-coref": "A co-reference to a KPI mentioned in a previous sentence.",
#     "false-positive": "Captures tokens that are similar to other entities,"
#     " but are explicitly not one of them, e.g. when the writer of the report forecasts next year’s revenue.",
# }
TAG_DICT = {
    "kpi": "Key Performance Indicators expressible in numerical and monetary value",
    "cy": "Current Year monetary value",
    "py": "Prior Year monetary value",
    "py1": "Two Year Past Value",
    # "increase": "",
    # "increase-py": "",
    # "decrease": "",
    # "decrease-py": "",
    # "thereof": "",
    # "attr": "",
    # "kpi-coref": "",
    # "false-positive": ""
}
TAG_PAREN_RE = (r"\[", r"\]")
TAG_PAREN = tuple((e.strip("\\") for e in TAG_PAREN_RE))
TAG_PAREN_ESC = ("(", ")")
UNUSED_SPLIT = "unused"
SPLIT_DICT = {TRAIN_SPLIT: "train", VALID_SPLIT: "valid", TEST_SPLIT: "test", UNUSED_SPLIT: None}
SPLIT_TYPE_DICT = {v: k for (k, v) in SPLIT_DICT.items()}


class KPIEDGARScenario(Scenario):
    """
    Paper:
    T. Deußer et al.,
    “KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents.” 2022.
    https://arxiv.org/abs/2210.09163

    Website:
    https://github.com/tobideusser/kpi-edgar

    This is a dataset for Named Entity Recognition task for financial domain.

    Concretely, we prompt models using the following format:

    ```
Context: {Sentence}
Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets. 
kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
Answer:
    ```

    Example

    ```
Context: The following table summarizes our total share-based compensation expense and excess tax benefits recognized : As of December 28 , 2019 , there was $ 284 million of total unrecognized compensation cost related to nonvested share-based compensation grants .
Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.
kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
Answer: 
    ```

    Reference:
    ```
    284 [cy], total unrecognized compensation cost [kpi]
    ```

    """  # noqa

    name = "kpi_edgar"
    description = "Named Entity Recognition from financial documents."
    tags = ["question_answering"]  # TODO? https://crfm-helm.readthedocs.io/en/latest/schemas/
    base_url = "https://github.com/tobideusser/kpi-edgar/raw/main/data/kpi_edgar.json"
    dataset_file_name = "kpi_edgar.json"

    is_extraction = True

    @staticmethod
    def extract_samples(dataset_obj: List[Dict]) -> List[Dict]:
        def get_list(x: Any) -> List[Dict]:
            return x if isinstance(x, list) else []

        doc_seg_sentence_list_list_list = [
            [[st for st in get_list(seg["sentences"])] for seg in get_list(doc["segments"])] for doc in dataset_obj
        ]
        seg_sentence_list_list = list(itertools.chain.from_iterable(doc_seg_sentence_list_list_list))
        sentence_list = list(itertools.chain.from_iterable(seg_sentence_list_list))
        return sentence_list

    @staticmethod
    def get_split(sample_dict: Dict, split_type_dict: Dict[Any, str]) -> str:
        return split_type_dict[sample_dict["split_type"]]

    @staticmethod
    def insert_tags(word_list: List[str], anno_list: List[Dict]) -> List[str]:
        def add_tag_part(the_word: str, tag_type: str, is_start: bool) -> str:
            tagged_word = ("<%s>" % tag_type) + the_word if is_start else the_word + ("</%s>" % tag_type)
            return tagged_word

        curr_word_list = list(word_list)
        for anno in anno_list:
            start_idx = anno["start"]
            end_idx = anno["end"] - 1
            if start_idx >= len(curr_word_list):
                logging.warning(curr_word_list)
                logging.warning(start_idx)
            if end_idx >= len(curr_word_list):
                logging.warning(curr_word_list)
                logging.warning(end_idx)

            curr_word_list[start_idx] = add_tag_part(curr_word_list[start_idx], anno["type_"], True)
            curr_word_list[end_idx] = add_tag_part(curr_word_list[end_idx], anno["type_"], False)

        return curr_word_list

    @staticmethod
    def create_prompt(sample: dict) -> Tuple[str, str]:
        word_list = [wd["value"] for wd in sample["words"]]
        anno_list = sample["entities_anno"]
        tag_dict = TAG_DICT
        tag_desc_list = ["<%s></%s>: %s" % (key, key, val) for (key, val) in tag_dict.items()]
        tag_desc = "\n".join(tag_desc_list)

        passage = " ".join(word_list)
        context = "Context: %s\n" % (passage)
        question = "Question: Enclose KPIs (key performance indicators) and values of those in the above text with the following tags.\n"  # noqa
        tags = "%s" % tag_desc
        prompt = context + question + tags

        tagged_word_list = KPIEDGARScenario.insert_tags(word_list, anno_list)
        answer = " ".join(tagged_word_list)
        return (prompt, answer)

    @staticmethod
    def escape_parenthesis(text: str, re_tag_paren: Tuple[str, str], esc_paren: Tuple[str, str]) -> str:
        tmp0 = re.sub(re_tag_paren[0], esc_paren[0], text)
        tmp1 = re.sub(re_tag_paren[1], esc_paren[1], tmp0)
        return tmp1

    @staticmethod
    def create_ans_list_extraction(
        word_list: List[str],
        anno_list: List[Dict],
    ) -> List[str]:
        def create_one_ans(word_list: List[str], anno: Dict):
            start_idx = anno["start"]
            end_idx = anno["end"]
            anno_word_list = word_list[start_idx:end_idx]
            tmp_phrase = " ".join(anno_word_list)
            phrase = KPIEDGARScenario.escape_parenthesis(tmp_phrase, TAG_PAREN_RE, TAG_PAREN_ESC)
            ans_str = "%s %s%s%s" % (phrase, TAG_PAREN[0], anno["type_"], TAG_PAREN[1])
            return ans_str

        ans_list = [create_one_ans(word_list, anno) for anno in anno_list]
        return ans_list

    @staticmethod
    def create_prompt_extraction(sample: dict) -> Tuple[str, str]:
        word_list = [wd["value"] for wd in sample["words"]]
        anno_list = sample["entities_anno"]
        tag_dict = TAG_DICT
        tag_desc_list = ["%s: %s" % (key, val) for (key, val) in tag_dict.items()]
        tag_desc = ", ".join(tag_desc_list) + "."

        passage0 = " ".join(word_list)
        passage = KPIEDGARScenario.escape_parenthesis(passage0, TAG_PAREN_RE, TAG_PAREN_ESC)
        context = "Context: %s\n" % (passage)
        question = "Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.\n"  # noqa
        tags = "%s" % tag_desc
        prompt = context + question + tags

        ans_list = KPIEDGARScenario.create_ans_list_extraction(word_list, anno_list)
        answer = ", ".join(ans_list)
        return (prompt, answer)

    @staticmethod
    def get_split_instances(dataset_obj: List[Dict], is_extraction=True) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            dataset_obj (list): Dataset for the corresponding data split

        Returns:
            List[Instance]: Instances for the specified split
        """
        sample_list = KPIEDGARScenario.extract_samples(dataset_obj)

        prompt_split_list = [
            (
                KPIEDGARScenario.create_prompt(sample)
                if not is_extraction
                else KPIEDGARScenario.create_prompt_extraction(sample),
                KPIEDGARScenario.get_split(sample, SPLIT_TYPE_DICT),
            )
            for sample in sample_list
        ]

        instance_list = [
            Instance(
                input=Input(text=pr),
                references=[Reference(Output(text=ans), tags=[CORRECT_TAG])],
                split=split,
            )
            for ((pr, ans), split) in prompt_split_list
        ]

        return instance_list

    def get_instances(self, output_path: str) -> List[Instance]:

        random.seed(0)  # we pick a random dialogue point to query the model

        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        base_url = self.base_url
        dataset_file_name = self.dataset_file_name
        target_path = os.path.join(data_path, dataset_file_name)
        ensure_file_downloaded(
            source_url=base_url,
            target_path=target_path,
            unpack=False,
        )

        instances = []
        with open(target_path, "r") as f:
            dataset_dict = json.load(f)
            instances = KPIEDGARScenario.get_split_instances(dataset_dict, self.is_extraction)
        return instances
