from collections import namedtuple
import random
from typing import List, Tuple

from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG

Relation = namedtuple("Relation", ["A", "B"])


class NumeracyScenario(Scenario):
    """
    A silly task for debugging the infrastructure where the input is a random
    sequence of tokens and the output is one of those tokens.

    Example:

        2 5 3 -> 2
        1 4 0 -> 0
    """

    name = "numeracy"
    description = "A simple scenario"
    tags = ["simple"]

    def __init__(
        self,
        num_in_context_samples: int,
        relation_size: int,
        num_train_instances: int,
        num_test_instances: int,
        delimiter: str = ", ",
    ):
        self.num_in_context_samples = num_in_context_samples
        self.relation_size = relation_size
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.delimiter = delimiter

    def get_instances(self) -> List[Instance]:
        random.seed(1)

        def generate_rel() -> Relation:
            """An input is just a random sequence of tokens (e.g., 4 2 5 6)."""
            A = random.randint(0, 4)
            B = random.randint(-4, 4)
            return Relation(A=A, B=B)

        def generate_datapoint(rel: Relation) -> Tuple[str, str]:
            x = random.randint(0, 99)
            y = rel.A * x + rel.B
            return (str(x), str(y))

        def generate_instance(num_in_context_samples: int, tags: List[str]):
            """Generate a random instance with `tags`."""
            rel = generate_rel()
            dataset = [generate_datapoint(rel) for _ in range(num_in_context_samples)]

            inputs = [self.delimiter.join(xy) for xy in dataset[:-1]] + [dataset[-1][0]]
            input = "\n".join(inputs) + self.delimiter.rstrip()  # TODO whitespace
            output = dataset[-1][-1]
            references = [
                Reference(output=output, tags=[CORRECT_TAG]),  # Correct output
            ]
            return Instance(input=input, references=references, tags=tags)

        def generate_instances(num_instances: int, tags: List[str]):
            return [generate_instance(self.num_in_context_samples, tags) for _ in range(num_instances)]

        return generate_instances(self.num_train_instances, [TRAIN_TAG]) + generate_instances(
            self.num_test_instances, [TEST_TAG]
        )
