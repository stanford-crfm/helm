import os
import re
from typing import List, Any, Dict
import json

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from .scenario import Scenario, Instance, Input, Output, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG


_DATASET_URL_PREFIX = "https://github.com/czyssrs/FinQA/raw/0f16e2867befa6840783e58be38c9efb9229d742/dataset/"


class FinQAScenario(Scenario):
    """
    FinQA is a question answering task over financial reports that requires robust numerical reasoning.

    FinQA: A Dataset of Numerical Reasoning over Financial Data
    https://arxiv.org/abs/2109.00122
    https://github.com/czyssrs/FinQA



    We add the sub-headers "Pre-table text", "Table in JSON format", "Post-table text" to the input. Example:

    ```
    Pre-table text: printing papers net sales for 2006 decreased 3% ( 3 % ) from both 2005 and 2004 due principally to the sale of the u.s .
    coated papers business in august 2006 .
    [more lines]

    Table in JSON format: [["in millions", "2006", "2005", "2004"], ["sales", "$ 6930", "$ 7170", "$ 7135"], ["operating profit", "$ 677", "$ 473", "$ 508"]]

    Post-table text: u.s .
    uncoated papers net sales in 2006 were $ 3.5 billion , compared with $ 3.2 billion in 2005 and $ 3.3 billion in 2004 .
    [more lines]

    Question: brazilian paper sales represented what percentage of printing papers in 2005?
    Answer:
    ```
    """

    name = "fin_qa"
    description = "FinQA"
    tags = ["question_answering", "financial"]

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        # Note: only train and test splits are used; dev split is not used
        instances: List[Instance] = []
        for split in [TRAIN_SPLIT, TEST_SPLIT]:
            file_name = f"{split}.json"
            target_path = os.path.join(data_path, file_name)
            ensure_file_downloaded(
                source_url=_DATASET_URL_PREFIX + file_name,
                target_path=target_path,
            )
            with open(target_path, "r") as f:
                rows = json.load(f)
                for row in rows:
                    pre_text = "Pre-table text: " + "\n".join(row["pre_text"])
                    post_text = "Post-table text: " + "\n".join(row["post_text"])
                    table = "Table in JSON format: " + json.dumps(row["table"])
                    question = "Question: " + row["qa"]["question"]
                    text = "\n\n".join([pre_text, table, post_text, question])
                    reference = Reference(
                        Output(text=str(row["qa"]["exe_ans"])),
                        tags=[CORRECT_TAG],
                    )
                    instance: Instance = Instance(
                        input=Input(text=text),
                        references=[reference],
                        split=split,
                    )
                    instances.append(instance)
        return instances
