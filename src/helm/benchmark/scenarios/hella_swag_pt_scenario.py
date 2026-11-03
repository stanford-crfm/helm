from typing import Any, List
from pathlib import Path
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


class HellaSwag_PT_Scenario(Scenario):
    """
    Multiple-choice commonsense reasoning using Multilingual HellaSwag.

    Each example contains a context and four candidate continuations.
    The task is to select the most plausible continuation.
    """

    name = "hellaswag_pt"
    description = "Multilingual HellaSwag: select the most plausible continuation of a context"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:

        dataset: Any
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset("alexandrainst/m_hellaswag", "pt", split="val", cache_dir=cache_dir)

        for example in dataset:

            example_id = example["id"]
            context = example["ctx"]
            endings = example["endings"]

            if not isinstance(context, str) or not context.strip():
                raise ValueError(f"{example_id}: empty or invalid context")

            if any(not isinstance(text, str) or not text.strip() for text in endings):
                raise ValueError(f"{example_id}: empty or invalid alternative")

            try:
                correct_index = int(example["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{example_id} invalid answer label") from exc

            if not 0 <= correct_index < len(endings):
                raise ValueError(f"{example_id}: answer label out of range")

            references = [
                Reference(
                    output=Output(text=text),
                    tags=[CORRECT_TAG] if index == correct_index else [],
                )
                for index, text in enumerate(endings)
            ]

            instances.append(
                Instance(
                    input=Input(text=context),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
