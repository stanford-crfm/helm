import csv
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class ArabicToolUsageScenario(Scenario):

    name = "arabic_tool_usage"
    description = "Arabic tool usage"
    tags = ["tool_usage"]

    def __init__(self, lang: str):
        super().__init__()
        self.lang = lang
        if self.lang == "en":
            self.action_prompt = "action_prompt"
        elif self.lang == "ar":
            self.action_prompt = "action_prompt_ar"
        else:
            raise Exception(f"Unsupported language {lang}")

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        csv_path = "/home/yifanmai/oss/helm/arabic-bench/tool_usage/metatool_multi_prompts_250_ar.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input = Input(row[self.action_prompt])
                references = [Reference(output=Output(text=row["gold_tools"]), tags=[CORRECT_TAG])]
                instances.append(
                    Instance(
                        input=input,
                        references=references,
                        split=TEST_SPLIT,
                    )
                )

        return instances
