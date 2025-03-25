import json
import os
from typing import Dict, List, Any

from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Output,
)
from helm.common.general import ensure_file_downloaded


class ConvFinQACalcScenario(Scenario):
    """A mathematical calculation benchmark based on ConvFinQA.

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

    DATASET_DOWNLOAD_URL: str = (
        "https://github.com/czyssrs/ConvFinQA/raw/cf3eed2d5984960bf06bb8145bcea5e80b0222a6/data.zip"
    )

    _SPLIT_TO_JSON_FILE_NAME: Dict[str, str] = {TRAIN_SPLIT: "train_turn.json", VALID_SPLIT: "dev_turn.json"}

    def make_pseudo_markdown_table(self, table: List[List[Any]], sep: str = "\n") -> str:
        markdown_lines: List[str] = []

        for row in table:
            row_inner_markdown = " | ".join([str(cell) for cell in row])
            row_markdown = f"| {row_inner_markdown} |"
            markdown_lines.append(row_markdown)

        return sep.join(markdown_lines)

    def convert_to_instance(self, dic: Dict[str, Any], split: str, sep: str = "\n") -> Instance:
        linearized_table = self.make_pseudo_markdown_table(dic["table"])
        input_text = f"Table: {sep}{linearized_table}{sep}{sep}"

        if "gold_ind" in dic["annotation"]:
            facts = dic["annotation"]["gold_ind"]
        elif "gold_inds" in dic["annotation"]:
            facts = dic["annotation"]["gold_inds"]
        else:
            facts = {}
        table_text = ""
        for fact_type, fact in facts.items():
            if "text" in fact_type:
                table_text += fact
        if table_text:
            input_text += f"Text: {sep}{table_text}{sep}{sep}"

        for ind, q in enumerate(dic["annotation"]["cur_dial"]):
            if ind < len(dic["annotation"]["cur_dial"]) - 1:
                input_text += f"Question: {q}{sep}Answer: {dic['annotation']['exe_ans_list'][ind]}{sep}"
            else:
                input_text += f"Question: {q}"

        answer = str(dic["annotation"]["exe_ans"])
        return Instance(
            input=Input(text=input_text),
            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
            split=split,
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=os.path.join(output_path, "data"),
            unpack=True,
            unpack_type="unzip",
        )
        instances: List[Instance] = []
        for split, json_file_name in self._SPLIT_TO_JSON_FILE_NAME.items():
            json_file_path = os.path.join(data_path, json_file_name)
            with open(json_file_path) as f:
                raw_instances = json.load(f)
                for raw_instance in raw_instances:
                    instances.append(self.convert_to_instance(raw_instance, split))
        return instances
