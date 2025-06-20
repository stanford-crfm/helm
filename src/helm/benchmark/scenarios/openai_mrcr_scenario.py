import json
import os
import re
from typing import List, Optional

import datasets
import tiktoken

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Output,
    Reference,
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)
from helm.common.general import ensure_directory_exists


class OpenAIMRCRScenario(Scenario):
    """OpenAI MRCR scenario

    OpenAI MRCR (Multi-round co-reference resolution) is a long context dataset for benchmarking
    an LLM's ability to distinguish between multiple needles hidden in context. This eval is
    inspired by the MRCR eval first introduced by Gemini (https://arxiv.org/pdf/2409.12640v2).

    The task is as follows: The model is given a long, multi-turn, synthetically generated
    conversation between user and model where the user asks for a piece of writing about a topic,
    e.g. "write a poem about tapirs" or "write a blog post about rocks". Hidden in this conversation
    are 2, 4, or 8 identical asks, and the model is ultimately prompted to return the i-th instance
    of one of those asks. For example, "Return the 2nd poem about tapirs".

    Reference: https://huggingface.co/datasets/openai/mrcr"""

    name = "openai_mrcr"
    description = "OpenAI MRCR (Multi-round co-reference resolution) is a long context dataset for benchmarking an LLM's ability to distinguish between multiple needles hidden in context. This eval is inspired by the MRCR eval first introduced by [Vodrahalli et al., 2024](https://arxiv.org/pdf/2409.12640v2)."  # noqa: E501
    tags = ["long_context", "mrcr"]

    NEEDLES_OPTIONS = [2, 4, 8]

    def __init__(self, needles: int, max_num_words: Optional[int] = None):
        super().__init__()
        self.needles = needles
        self.max_num_words = max_num_words
        if needles not in self.NEEDLES_OPTIONS:
            raise Exception(f"Needles must be one of {self.NEEDLES_OPTIONS}")
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def count_words(self, messages: list[dict]) -> int:
        return sum([len(re.split(r"\s+", m["content"].strip())) for m in messages])

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "openai/mrcr",
            cache_dir=cache_dir,
            split="train",
            data_files=[f"{self.needles}needle.parquet"],
            revision="204b0d4e8d9ca5c0a90bf942fdb2a5969094adc0",
        )
        instances = []
        for idx, row in enumerate(dataset):
            messages = json.loads(row["prompt"])
            if self.max_num_words and self.count_words(messages) > self.max_num_words:
                continue
            input = Input(messages=messages)
            references = [Reference(output=Output(text=row["answer"]), tags=[CORRECT_TAG])]
            instance = Instance(
                id=f"{self.needles}needle{idx}",
                input=input,
                references=references,
                split=TEST_SPLIT,
                extra_data={"random_string_to_prepend": row["random_string_to_prepend"]},
            )
            instances.append(instance)

        return instances
