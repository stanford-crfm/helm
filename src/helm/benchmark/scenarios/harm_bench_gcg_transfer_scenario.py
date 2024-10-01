import os
import pandas as pd
from typing import List

from helm.common.general import ensure_file_downloaded

from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class HarmBenchGCGTransferScenario(Scenario):
    """
    HarmBenchGCG-T is a standardized evaluation framework for automated red teaming.
    HarmBench identifies key considerations previously unaccounted for in red teaming
    evaluations and systematically designed prompts that meet these criteria.

    https://arxiv.org/abs/2402.04249
    """

    name = "harm_bench_gcg_transfer"
    description = "HarmBench GCG-T dataset - standard and contextual attacks"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        os.makedirs(data_path, exist_ok=True)
        url = "https://raw.githubusercontent.com/farzaank/harmbench-gcg-ensembled/refs/heads/main/output.csv"  # noqa: E501
        outf_path = os.path.join(data_path, "harmbench_gcg.csv")
        ensure_file_downloaded(
            source_url=url,
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
            references = [Reference(output=Output(text="unsafe behavior"), tags=[])]
            input_text = row["behavior_prompt"]
            input = Input(text=input_text)
            id = row["behavior_name"]
            instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            instances.append(instance)
        return instances
