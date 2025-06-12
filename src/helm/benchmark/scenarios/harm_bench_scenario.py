import os
import pandas as pd
from typing import List

from helm.common.general import ensure_file_downloaded

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class HarmBenchScenario(Scenario):
    """
    HarmBench is a standardized evaluation framework for automated red teaming.
    HarmBench identifies key considerations previously unaccounted for in red teaming
    evaluations and systematically designed prompts that meet these criteria.

    https://arxiv.org/abs/2402.04249
    """

    name = "harm_bench"
    description = "HarmBench dataset - standard and contextual attacks"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        os.makedirs(data_path, exist_ok=True)
        url_prefix = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/"  # noqa: E501
        outf_path = os.path.join(data_path, "harmbench_behaviors_text_all.csv")
        ensure_file_downloaded(
            source_url=url_prefix + "behavior_datasets/harmbench_behaviors_text_all.csv",
            target_path=outf_path,
            unpack=False,
        )

        df = pd.read_csv(outf_path)

        # Read all the instances
        instances: List[Instance] = []

        # we take the direct_request template from HarmBench for simplicity
        # see the example here
        # https://github.com/centerforaisafety/HarmBench/blob/main/baselines/direct_request/direct_request.py
        #
        # in the future we can use other types of templates including various red team attacks
        for i, row in df.iterrows():
            if row["FunctionalCategory"] == "contextual":
                context = row["ContextString"]
                input_text = context + "\n\n---\n\n" + row["Behavior"]
            else:
                input_text = row["Behavior"]
            input = Input(text=input_text)
            tag = row["Tags"]
            id = row["BehaviorID"]
            references = [Reference(output=Output(text=row["SemanticCategory"]), tags=[])]
            if tag != "":
                instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            else:
                instance = Instance(input=input, split=TEST_SPLIT, references=references, sub_split=tag, id=id)
            instances.append(instance)
        return instances
