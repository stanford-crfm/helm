import string
import json
import pandas as pd
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.run_specs.medhelm.benchmark_config import get_benchmark_config_from_path
from helm.common.general import check_file_exists

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    CORRECT_TAG,
    Reference,
    Input,
    Output,
    TEST_SPLIT,
    ScenarioMetadata,
)


class MedHELMConfigurableScenario(Scenario):
    """
    MedHELM configuratble scenario
    """

    tags = ["biomedical"]

    def __init__(self, name: str, config_path: str):
        super().__init__()
        self.benchmark_config = get_benchmark_config_from_path(config_path)
        self.name = name
        self.description = self.benchmark_config.description

    def get_columns_in_template(self, template: str) -> List[str]:
        """
        Extract field names from a template string using Python's Formatter.
        Example: "Name: {name}, Age: {age}" → ["name", "age"]
        """
        formatter = string.Formatter()
        return [fname for _, fname, _, _ in formatter.parse(template) if fname]

    def populate_template(self, template: str, row: pd.Series, fields: List[str]) -> str:
        """
        Populate the template with values from the row using format_map.
        Missing fields default to empty string.
        """
        mapping = {field: row.get(field, "") for field in fields}
        return template.format_map(mapping)

    def get_references(self, row: pd.Series) -> List[Reference]:
        references: List[Reference] = []
        if "correct_answer" in row:
            references.append(Reference(Output(text=row["correct_answer"]), tags=[CORRECT_TAG]))
        if "incorrect_answers" in row:
            for incorrect_answer in row["incorrect_answers"]:
                references.append(Reference(Output(text=incorrect_answer), tags=[]))
        return references

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.benchmark_config.prompt_file, msg=f"Prompt file for {self.name} does not exist")
        check_file_exists(self.benchmark_config.dataset_file, msg=f"Dataset file for {self.name} does not exist")
        instances: List[Instance] = []
        df = pd.read_csv(self.benchmark_config.dataset_file)
        if "correct_answer" not in df.columns:
            if not self._is_llm_as_judge() or len(self.benchmark_config.metrics) > 1:
                raise ValueError(
                    "Dataset must contain 'correct_answer' column unless using jury_score as the only metric."
                )
        if "incorrect_answers" in df.columns:
            df["incorrect_answers"] = df["incorrect_answers"].apply(json.loads)
        with open(self.benchmark_config.prompt_file, "r") as f:
            template = f.read()
        fields = self.get_columns_in_template(template)
        for _, row in df.iterrows():
            filled = self.populate_template(template, row, fields)
            prompt = Input(text=filled)
            instances.append(Instance(input=prompt, references=self.get_references(row), split=TEST_SPLIT))
        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name=self.name,
            display_name=self.name,
            description=self.description,
            taxonomy=TaxonomyInfo(
                task="",
                what="",
                when="",
                who="",
                language="",
            ),
            main_metric=self.benchmark_config.main_metric.name,
            main_split="test",
        )

    def _is_llm_as_judge(self) -> bool:
        for metric in self.benchmark_config.metrics:
            if metric.name == "jury_score":
                return True
        return False
