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


class ARC_Challenge_PT_Scenario(Scenario):
    """
    Multiple-choice science reasoning using the Portuguese subset of Multilingual ARC.

    Each example contains a question, three to five candidate answers, and the
    letter of the correct answer.
    """

    name = "arc_challenge_pt"
    description = "Multilingual ARC-Challenge: answer science questions in Portuguese"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:

        dataset: Any
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset("alexandrainst/m_arc", "pt", split="test", cache_dir=cache_dir)

        for example in dataset:

            example_id = example["id"]
            question = example["instruction"]
            answer = example["answer"]

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{example_id}: empty or invalid question")

            choices = []
            for label in "ABCDE":
                text = example.get(f"option_{label.lower()}")
                if text is None:
                    continue
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{example_id}: empty or invalid alternative {label}")
                choices.append((label, text))

            if len(choices) < 2:
                raise ValueError(f"{example_id}: expected at least two alternatives")

            if not isinstance(answer, str) or answer not in [label for label, _ in choices]:
                raise ValueError(f"{example_id}: invalid answer label")

            references = [
                Reference(
                    output=Output(text=text),
                    tags=[CORRECT_TAG] if label == answer else [],
                )
                for label, text in choices
            ]

            instances.append(
                Instance(
                    input=Input(text=question),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
