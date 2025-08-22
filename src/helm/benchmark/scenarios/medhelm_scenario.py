import re
import pandas as pd
from typing import List

from helm.benchmark.run_specs.medhelm_run_specs import BenchmarkConfig
from helm.common.general import check_file_exists

from helm.benchmark.scenarios.scenario import Scenario, Instance, CORRECT_TAG, Reference, Input, Output, TEST_SPLIT


class MedHELMScenario(Scenario):
    """
    MedHELM resuable scenario
    """

    tags = ["biomedical"]

    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__()
        self.benchmark_config = benchmark_config
        self.name = benchmark_config.name
        self.description = benchmark_config.description

    def get_columns_in_template(self, template: str) -> List[str]:
        fields = re.findall(r"\{\{(.*?)\}\}", template)
        return fields

    def populate_template(self, template: str, row: pd.Series, fields: List[str]) -> str:
        filled = template
        for field in fields:
            if field in row.index:
                filled = filled.replace(f"{{{{{field}}}}}", str(row[field]))
        return filled

    def get_references(self, row: pd.Series) -> List[Reference]:
        references: List[Reference] = [Reference(Output(text=row["correct_answer"]), tags=[CORRECT_TAG])]
        if "incorrect_answers" in row:
            for incorrect_answer in row["incorrect_answers"]:
                references.append(Reference(Output(text=incorrect_answer), tags=[]))
        return references

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.benchmark_config.prompt_file, msg=f"Prompt file for {self.name} does not exist")
        check_file_exists(self.benchmark_config.dataset_file, msg=f"Dataset file for {self.name} does not exist")
        instances: List[Instance] = []
        df = pd.read_csv(self.benchmark_config.dataset_file)
        with open(self.benchmark_config.prompt_file, "r") as f:
            template = f.read()
        if "incorrect_answers" in df.columns:
            df["incorrect_answers"] = df["incorrect_answers"].apply(lambda x: x.split(","))

        fields = self.get_columns_in_template(template)
        for _, row in df.iterrows():
            filled = self.populate_template(template, row, fields)
            prompt = Input(text=filled)
            instances.append(Instance(input=prompt, references=self.get_references(row), split=TEST_SPLIT))
        return instances
