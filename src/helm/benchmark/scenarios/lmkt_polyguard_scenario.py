"""A high quality multilingual benchmark with 29K samples for the evaluation of safety guardrails."""

from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)

SUPPORTED_LANGUAGES = ["ar", "cs", "de", "en", "es", "hi", "it", "ja", "ko", "nl", "pl", "pt", "ru", "sv", "zh", "th"]
CODE_MAP = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "zh": "Chinese",
    "th": "Thai",
}
REQUEST_TYPES = ["harmful", "unharmful", "both"]


class CulturalSafetyApplicationPolyGuardScenario(Scenario):
    """A high quality multilingual benchmark with 29K samples for the evaluation of safety guardrails."""

    name = "cultural_safety_application_polyguard"
    description = "Evaluates the safety of LLMs"
    tags = ["lmkt", "cultural_safety_application"]

    def __init__(self, language: str, request_type: str):
        super().__init__()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported languages are: {SUPPORTED_LANGUAGES}")
        if request_type not in REQUEST_TYPES:
            raise ValueError(f"{request_type} not found. Request types are: {REQUEST_TYPES}")

        self.language = CODE_MAP[language]
        self.request_type = request_type

    def get_instances(self, output_path: str) -> List[Instance]:

        instances: List[Instance] = []

        dataset = load_dataset(
            "ToxicityPrompts/PolyGuardPrompts",
            trust_remote_code=True,
            revision="c5b466a95b64ff121db4398246b6abb7672696ec",
            split="test",
        )
        if self.request_type != "both":
            dataset = dataset.filter(
                lambda example: example["language"] == self.language
                and (example["prompt_harm_label"] == self.request_type)
            )
        else:
            dataset = dataset.filter(lambda example: example["language"] == self.language)
        # Create instances for each selected question variant

        for _, row in enumerate(dataset):
            input = Input(text=row["prompt"].strip())
            instance = Instance(
                input=input,
                references=[],
                split=TEST_SPLIT,
                extra_data={
                    "prompt_harm_label": row["prompt_harm_label"],
                    "subcategory": row["subcategory"],
                    "language": self.language,
                },
            )
            instances.append(instance)

        return instances
