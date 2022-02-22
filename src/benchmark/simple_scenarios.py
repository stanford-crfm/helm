import random
from typing import List

from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG


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
        self.num_input_tokens = num_input_tokens
        self.vocab_size = vocab_size
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances

    def get_instances(self) -> List[Instance]:
        random.seed(1)

        def generate_seq() -> List[str]:
            """An input is just a random sequence of tokens (e.g., 4 2 5 6)."""
            return [str(random.randint(0, self.vocab_size - 1)) for _ in range(self.num_input_tokens)]

        def generate_instance(tags: List[str]):
            """Generate a random instance with `tags`."""
            tokens = generate_seq()
            input = " ".join(tokens)
            output = random.choice(tokens)
            references = [
                Reference(output=output, tags=[CORRECT_TAG]),  # Correct output
                Reference(output="-1", tags=[]),  # Wrong output
            ]
            return Instance(input=input, references=references, tags=tags)

        def generate_instances(num_instances: int, tags: List[str]):
            return [generate_instance(tags) for _ in range(num_instances)]

        return generate_instances(self.num_train_instances, [TRAIN_TAG]) + generate_instances(
            self.num_test_instances, [TEST_TAG]
        )
