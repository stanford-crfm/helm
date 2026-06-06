"""A multilingual closed-book QA (CBQA) dataset that Evaluates
Cross-Lingual Knowledge Transfer in a simple, black-box manner"""

import os
from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)

SUPPORTED_LANGUAGES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "zh": "Chinese",
}


class CulturalKnowledgeRememberingEclekticScenario(Scenario):
    """Cultural values understanding evaluation based on Vietnam World Values Survey responses."""

    name = "cultural_knowledge_remembering_eclektic"
    description = "Evaluates the cross-lingual knowledge transfer ability of LLMs"
    tags = ["lmkt", "cultural_knowledge_remembering"]

    def get_instances(self, output_path: str) -> List[Instance]:

        instances: List[Instance] = []

        dataset = load_dataset(
            "ura-hcmut/ECLeKTic",
            data_files="eclektic_main.jsonl",
            cache_dir=os.path.join(output_path, "data"),
            trust_remote_code=True,
            revision="86650a00986420df9939b5f29d256aebad04c767",
            split="train",
        )

        # Create instances for each selected question variant
        for _, row in enumerate(dataset):
            for lang in SUPPORTED_LANGUAGES:
                new_ex = {
                    "original_lang": row["original_lang"],
                    "lang": lang,
                    "title": row["title"],
                    "url": row["url"],
                    "orig_content": row["content"],
                    "orig_question": row["question"],
                    "orig_answer": row["answer"],
                    "content": row[f"{lang}_c"],
                }

                input = Input(text=row[f"{lang}_q"].strip())
                reference = Reference(Output(text=row[f"{lang}_a"]), tags=[CORRECT_TAG])
                instance = Instance(
                    input=input,
                    references=[reference],
                    split=TEST_SPLIT,
                    extra_data=new_ex,
                )
                instances.append(instance)

        return instances
