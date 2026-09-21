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


class BLUEXScenario(Scenario):
    """
    The BLUEX dataset is a benchmark used for evaluating natural language processing models in Brazilian Portuguese.
    It consists of multiple-choice questions taken from official entrance exams of Unicamp (Convest) and USP (Fuvest),
    covering various high school subjects. The questions include both textual prompts and visual elements. This dataset
    was developed to assess the performance of models on tasks involving comprehension and reasoning, with a specific
    focus on texts and exams originally written in Portuguese.
    """

    name = "bluex"
    description = "MQA benchmark with questions from Brazilian entrance exams"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data and read all the dialogues
        dataset: Any
        # Read all the instances
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset(
            "portuguese-benchmark-datasets/BLUEX",
            revision="d99cf6d05b50db7c42a605e5e2924cbd46f076c7",
            cache_dir=cache_dir,
        )
        for example in dataset["questions"]:
            # This scenario disregards issues with images
            if example["has_associated_images"]:
                continue
            question = example["question"]
            choices = example["alternatives"]
            answer = example["answer"]

            answers_dict = {}
            for alt in choices:
                if ")" in alt:
                    label, text = alt.split(")", 1)
                    label = label.strip().upper()
                    text = text.strip()
                    answers_dict[label] = text

            if answer not in answers_dict:
                continue

            correct_answer = answers_dict[answer]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

            instance = Instance(
                input=Input(text=question),
                split=TEST_SPLIT,
                references=[answer_to_reference(text) for text in answers_dict.values()],
            )
            instances.append(instance)
        return instances
