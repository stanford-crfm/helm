from typing import Any, List
import re
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


class HEALTHQA_BR_Scenario(Scenario):
    """
    HealthQA-BR is a large-scale benchmark designed to evaluate the clinical knowledge of Large Language Models (LLMs)
    within the Brazilian Unified Health System (SUS) context. It comprises 5,632 multiple-choice questions sourced from
    nationwide licensing exams and residency tests, reflecting real challenges faced by Brazil's public health sector.
    Unlike benchmarks focused on the U.S. medical landscape, HealthQA-BR targets the Brazilian healthcare ecosystem,
    covering a wide range of medical specialties and interdisciplinary professions such as nursing, dentistry,
    psychology, social work, pharmacy, and physiotherapy. This comprehensive approach enables a detailed assessment
    of AI models’ ability to collaborate effectively in the team-based patient care typical of SUS.
    """

    name = "healthqa_br"
    description = "MQA benchmark with questions from Brazilian entrance exams"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data and read all the dialogues
        dataset: Any
        # Read all the instances
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset("Larxel/healthqa-br", cache_dir=cache_dir)
        for example in dataset["train"]:
            question_choices = example["question"]
            answer = example["answer"].strip().upper()

            # Separate the question statement from the alternatives
            question_text, choices_text = self.split_question_and_choices(question_choices)

            # Extract alternatives from text choices_text
            pattern = r"'([A-Z])':\s*'([^']+)'"
            matches = re.findall(pattern, choices_text)
            answers_dict = {label: text for label, text in matches}

            if answer not in answers_dict:
                continue

            correct_answer_text = answers_dict[answer]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if correct_answer_text == answer else [])

            instance = Instance(
                input=Input(text=question_text),
                split=TEST_SPLIT,
                references=[answer_to_reference(text) for text in answers_dict.values()],
            )
            instances.append(instance)
        return instances

    def split_question_and_choices(self, full_text: str):
        # Search for the first occurrence of the alternative pattern
        match = re.search(r"\n'[A-Z]':\s*'.+", full_text)
        if match:
            # Everything before the alternatives
            question_part = full_text[: match.start()].strip()
            # All of the alternatives (from match to end)
            choices_part = full_text[match.start() :].strip()
        else:
            # If you don't find a pattern, consider everything as a question, and no alternative.
            question_part = full_text.strip()
            choices_part = ""

        return question_part, choices_part
