import json
import os
import pandas as pd
from typing import List

from helm.common.general import ensure_file_downloaded

from .scenario import CORRECT_TAG, Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class AutobencherSafetyScenario(Scenario):
    """
    Autobencher safety scenario

    AutoBencher uses a language model to automatically search
    for datasets. AutoBencher Capabilities consists of question
    answering datasets for math, multilingual, and knowledge-intensive
    question answering created by AutoBencher.

    Paper: https://arxiv.org/abs/2407.08351
    """

    name = "autobencher_safety"
    description = "Autobencher safety consists of question answering datasets"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        os.makedirs(data_path, exist_ok=True)
        url = "https://raw.githubusercontent.com/farzaank/AutoBencher/refs/heads/main/safety_processing/process%20full%20dataset%20for%20mTurk/full_dataset.json"  # noqa: E501
        outf_path = os.path.join(data_path, "full_dataset.json")
        ensure_file_downloaded(
            source_url=url,
            target_path=outf_path,
            unpack=False,
        )

        json_data = json.loads(outf_path)
        df = pd.DataFrame(json_data)

        # Read all the instances
        instances: List[Instance] = []

        for i, row in df.iterrows():
            references = [Reference(output=Output(text=row["gold_answer"]), tags=[CORRECT_TAG])]
            input_text = row["question"]
            input = Input(text=input_text)
            id = str(row["category"]) + str(row["id"])
            instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            instances.append(instance)
        return instances
