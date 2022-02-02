from collections import namedtuple
import random
from typing import List, Tuple

from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG

Relation = namedtuple("Relation", ["A", "B"])


def generate_rel(seed: int) -> Relation:
    """An input is just a random sequence of tokens (e.g., 4 2 5 6)."""
    random.seed(seed)
    A = random.randint(0, 4)
    B = random.randint(-4, 4)
    return Relation(A=A, B=B)


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
        relation_size: int,
        num_train_instances: int,
        num_test_instances: int,
        seed: int = 1,
        delimiter: str = ", ",
    ):
        self.seed = seed
        self.relation_size = relation_size
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.delimiter = delimiter

    def get_instances(self) -> List[Instance]:
        rel = generate_rel(self.seed)

        def generate_datapoint(rel: Relation) -> Tuple[List[str], str]:
            x = random.randint(0, 99)
            y = rel.A * x + rel.B
            return [str(x)], str(y)

        def generate_instance(tags: List[str]):
            """Generate a random instance with `tags`."""
            xs, y = generate_datapoint(rel)
            input = self.delimiter.join(xs)
            output = y
            references = [
                Reference(output=output, tags=[CORRECT_TAG]),  # Correct output
            ]
            return Instance(input=input, references=references, tags=tags)

        def generate_instances(num_instances: int, tags: List[str]):
            return [generate_instance(tags) for _ in range(num_instances)]

        return generate_instances(self.num_train_instances, [TRAIN_TAG]) + generate_instances(
            self.num_test_instances, [TEST_TAG]
        )
