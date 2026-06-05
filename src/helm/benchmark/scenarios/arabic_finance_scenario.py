import os
from typing import Dict, List

import datasets

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


class ArabicFinanceScenario(Scenario):

    name = "arabic_finance"
    description = "Arabic Finance"
    tags = ["finance"]

    def __init__(self, task: str, lang: str):
        super().__init__()
        self.task = task
        self.lang = lang
        if self.lang == "en":
            self.question_key = "question"
            self.choice_key = "choice"
            self.ground_truth_key = "ground_truth"
        elif self.lang == "ar":
            self.question_key = "question_ar"
            self.choice_key = "choice_ar"
            self.ground_truth_key = "ground_truth_ar"
        else:
            raise Exception(f"Unsupported language {lang}")

    def get_references(self, row: Dict[str, str]) -> List[Reference]:
        raise NotImplementedError()

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "stanford-crfm/arabic-enterprise",
            "finance",
            cache_dir=cache_dir,
            split="test",
            revision="35e114eda2e3450e0e69cf6bda9d3a2f54bf6f26",
        )
        assert isinstance(dataset, datasets.Dataset)

        instances: List[Instance] = []
        for row in dataset:
            if row["task"] != self.task:
                continue
            input = Input(text=row[self.question_key])
            references = self.get_references(row)
            instances.append(
                Instance(
                    id=row["id"],
                    input=input,
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances


class ArabicFinanceBoolScenario(ArabicFinanceScenario):
    name = "arabic_finance_bool"
    description = "Arabic Finance Bool"
    tags = ["finance"]

    def __init__(self, lang: str):
        super().__init__(task="bool", lang=lang)

    def get_references(self, row: Dict[str, str]) -> List[Reference]:
        if float(row[self.ground_truth_key]) == 1.0:
            return [Reference(output=Output(text="نعم" if self.lang == "ar" else "Yes"), tags=[CORRECT_TAG])]
        elif float(row[self.ground_truth_key]) == 0.0:
            return [Reference(output=Output(text="لا" if self.lang == "ar" else "No"), tags=[CORRECT_TAG])]
        else:
            raise ValueError(f"Unknown ground truth value {row[self.ground_truth_key]}")

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="arabic_finance_bool",
            main_metric="quasi_exact_match",
            main_split="test",
            display_name="Finance Boolean Verification",
            short_display_name="Finance Bool",
            description="Given questions from English language finance textbooks that have been "
            "translated to Arabic using machine translation, the LLM is prompted to answer "
            "the questions. In the boolean verification setting, the LLM must decide if a "
            "given statement is true or false.",
            short_description=None,
            taxonomy=TaxonomyInfo(
                task="boolean question answering",
                what="finance questions",
                when="before 2026",
                who="finance professionals",
                language="Arabic",
            ),
        )


class ArabicFinanceMCQScenario(ArabicFinanceScenario):
    name = "arabic_finance_mcq"
    description = "Arabic Finance MCQ"
    tags = ["finance"]

    def __init__(self, lang: str):
        super().__init__(task="mcq", lang=lang)

    def get_references(self, row: Dict[str, str]) -> List[Reference]:
        correct_choice_letter = row[self.ground_truth_key]
        raw_choices = row[self.choice_key]
        references: List[Reference] = []
        for choice_index, raw_choice in enumerate(raw_choices.split("\n")):
            choice_letter = chr(choice_index + ord("A"))
            choice_prefix = f"{choice_letter}. "
            assert raw_choice.startswith(choice_prefix)
            choice_text = raw_choice.removeprefix(choice_prefix)
            references.append(
                Reference(
                    output=Output(text=choice_text),
                    tags=[CORRECT_TAG] if choice_letter == correct_choice_letter else [],
                )
            )
        assert len(references) == 3
        return references

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="arabic_finance_mcq",
            main_metric="exact_match",
            main_split="test",
            display_name="Finance Multiple Choice Question Answering",
            short_display_name="Finance MCQA",
            description="Given questions from English language finance textbooks that have been "
            "translated to Arabic using machine translation, the LLM is prompted to answer "
            "the questions. In the MCQA setting, the LLM must choose between three possible "
            "choices.",
            short_description=None,
            taxonomy=TaxonomyInfo(
                task="multiple-choice question answering",
                what="finance questions",
                when="before 2026",
                who="finance professionals",
                language="Arabic",
            ),
        )


class ArabicFinanceCalculationScenario(ArabicFinanceScenario):
    name = "arabic_finance_calculation"
    description = "Arabic Finance Calculation"
    tags = ["finance"]

    def __init__(self, lang: str):
        super().__init__(task="calcu", lang=lang)

    def get_references(self, row: Dict[str, str]) -> List[Reference]:
        return [Reference(output=Output(text=row[self.ground_truth_key]), tags=[CORRECT_TAG])]

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="arabic_finance_calculation",
            main_metric="calculation_accuracy",
            main_split="test",
            display_name="Finance Calculation",
            short_display_name=None,
            description="Given questions from English language finance textbooks that have been "
            "translated to Arabic using machine translation, the LLM is prompted to answer "
            "the questions. In the calculation setting, the LLM must perform mathematical "
            "calculations to produce a numeric answer.",
            short_description=None,
            taxonomy=TaxonomyInfo(
                task="calculation",
                what="finance questions",
                when="before 2026",
                who="finance professionals",
                language="Arabic",
            ),
        )
