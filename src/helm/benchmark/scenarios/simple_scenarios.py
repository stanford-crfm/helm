import random
from typing import List

from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


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

    def get_instances(self) -> List[Instance]:
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
