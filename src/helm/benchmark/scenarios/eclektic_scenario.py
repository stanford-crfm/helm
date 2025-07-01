"""Cultural alignment evaluation scenario based on Vietnam World Values Survey responses."""

import os
import json
import random
from typing import List
from datasets import load_dataset
from huggingface_hub import snapshot_download

from helm.common.hierarchical_logger import hlog, hwarn
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    PassageQuestionInput,
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


class EclekticScenario(Scenario):
    """Cultural values understanding evaluation based on Vietnam World Values Survey responses."""

    name = "Eclektic"
    description = "Evaluates the cross-lingual knowledge transfer ability of LLMs"
    tags = ["cross-lingual"]

    def get_instances(self, output_path: str) -> List[Instance]:

        instances: List[Instance] = []

        dataset = load_dataset(
            "ura-hcmut/ECLeKTic", data_files="eclektic_main.jsonl", trust_remote_code=True, split="train"
        )

        # Create instances for each selected question variant
        instances: List[Instance] = []
        for _, row in enumerate(dataset):
            for lang in SUPPORTED_LANGUAGES:
                new_ex = {
                    "q_id": row["q_id"],
                    "original_lang": row["original_lang"],
                    "lang": lang,
                    "title": row["title"],
                    "url": row["url"],
                    "orig_content": row["content"],
                    "orig_question": row["question"],
                    "orig_answer": row["answer"],
                    "question": row[f"{lang}_q"],
                    "answer": row[f"{lang}_a"],
                    "content": row[f"{lang}_c"],
                }

                input = Input(text=row[f"{lang}_q"].strip())
                instance = Instance(
                    input=input,
                    references=[],
                    split=TEST_SPLIT,
                    extra_data=new_ex,
                )
                instances.append(instance)

        return instances
