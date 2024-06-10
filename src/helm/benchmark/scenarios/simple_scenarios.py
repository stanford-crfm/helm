"""Simple scenarios for debugging and for tutorials.

NOTE: Typically, each scenario should be in its own file,
but these scenarios are placed in the same module for
tutorial purposes."""

import random
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class SimpleMCQAScenario(Scenario):
    """Simple multiple-choice question answering scenario for tutorials and debugging.

    The task is to answer questions about whether two-digit numbers are even or odd.

    Example:

        Answer the following questions with a single letter only.

        Question: Is 24 even or odd?
        A. Even
        B. Odd
        Answer: A"""

    name = "simple_mcqa"
    description = "Answer if two-digit numbers are even or odd."
    tags = ["question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for i in range(10, 100):
            # NOTE: For simplicity, the input text and reference output text
            # is the same for all instances.
            # However, for most question answering scenarios, the input text
            # and reference output text can vary between questions.
            input = Input(text=f"Is {i} even or odd?")
            references = [
                Reference(Output(text="Even"), tags=[CORRECT_TAG] if i % 2 == 0 else []),
                Reference(Output(text="Odd"), tags=[CORRECT_TAG] if i % 2 == 1 else []),
            ]
            split = TRAIN_SPLIT if i <= 20 else TEST_SPLIT
            instance = Instance(input=input, references=references, split=split)
            instances.append(instance)
        return instances


class SimpleShortAnswerQAScenario(Scenario):
    """Simple short answer question answering scenario for tutorials and debugging.

    The task is to answer questions about whether two-digit numbers are even or odd.

    Example:

        Answer the following questions with a single word only.

        Question: Is 24 even or odd?
        Answer: Even"""

    name = "simple_mcqa"
    description = "Answer if two-digit numbers are even or odd."
    tags = ["question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for i in range(10, 100):
            # NOTE: For simplicity, the input text and reference output text
            # is the same for all instances.
            # However, for most question answering scenarios, the input text
            # and reference output text can vary between questions.
            input = Input(text=f"Is {i} even or odd?")
            correct_answer = "Even" if i % 2 == 0 else "Odd"
            # NOTE: Unlike multiple-choice question answering, only the correct
            # references are needed for short-answer question answering.
            references = [
                Reference(Output(text=correct_answer), tags=[CORRECT_TAG]),
            ]
            split = TRAIN_SPLIT if i <= 20 else TEST_SPLIT
            instance = Instance(input=input, references=references, split=split)
            instances.append(instance)
        return instances


class SimpleClassificationScenario(Scenario):
    """Simple multiple-choice question answering scenario for tutorials and debugging.

    The task is to classify two-digit numbers as even or odd.

    Example:

        Classify the following numbers by their pairity. The classes are "Even" and "Odd".

        Number: 24
        Pairity: Even"""

    name = "simple_classification"
    description = "Classify numbers by pairity."
    tags = ["classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for i in range(10, 100):
            input = Input(text=str(i))
            # NOTE: For classification scenarios, the reference outputs should be the same
            # for all instances, and should include both correct and incorrect classes.
            # HELM only supports single-label classification. Exactly one reference
            # should have the CORRECT_TAG tag.
            references = [
                Reference(Output(text="Even"), tags=[CORRECT_TAG] if i % 2 == 0 else []),
                Reference(Output(text="Odd"), tags=[CORRECT_TAG] if i % 2 == 1 else []),
            ]
            split = TRAIN_SPLIT if i <= 20 else TEST_SPLIT
            instance = Instance(input=input, references=references, split=split)
            instances.append(instance)
        return instances


class Simple1Scenario(Scenario):
    """
    A silly task for debugging the infrastructure where the input is a random
    sequence of tokens and the output is one of those tokens.

    Example:

        2 5 3 -> 2
        1 4 0 -> 0
    """

    name = "simple1"
    description = "A simple scenario"
    tags = ["simple"]

    def __init__(self, num_input_tokens: int, vocab_size: int, num_train_instances: int, num_test_instances: int):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.vocab_size = vocab_size
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        random.seed(1)

        def generate_seq() -> List[str]:
            """An input is just a random sequence of tokens (e.g., 4 2 5 6)."""
            return [str(random.randint(0, self.vocab_size - 1)) for _ in range(self.num_input_tokens)]

        def generate_instance(split: str) -> Instance:
            """Generate a random instance with `tags`."""
            tokens: List[str] = generate_seq()
            input: str = " ".join(tokens)
            output: str = random.choice(tokens)
            references: List[Reference] = [
                Reference(Output(text=output), tags=[CORRECT_TAG]),  # Correct output
                Reference(Output(text="-1"), tags=[]),  # Wrong output
            ]
            return Instance(Input(text=input), references=references, split=split)

        def generate_instances(num_instances: int, split: str):
            return [generate_instance(split) for _ in range(num_instances)]

        return generate_instances(self.num_train_instances, TRAIN_SPLIT) + generate_instances(
            self.num_test_instances, TEST_SPLIT
        )
