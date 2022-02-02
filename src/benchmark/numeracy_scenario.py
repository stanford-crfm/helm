from itertools import combinations_with_replacement
from math import comb  # type: ignore
import numpy as np
import random
from typing import List, Optional, Tuple

from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG


def generate_terms(degree: int, num_variables: int) -> List[List[int]]:
    return sum(
        [
            list(map(lambda _: list(_), combinations_with_replacement(range(num_variables), d)))
            for d in reversed(range(degree + 1))
        ],
        [],
    )


class PolyEq:
    def __init__(self, degree: int, num_variables: int, coeffs: List[int]):
        self.degree = degree
        self.num_variables = num_variables
        self.terms = generate_terms(degree, num_variables)
        self.coeffs = np.array(coeffs)

    def eval(self, vals: List[int]):
        return np.dot(self.coeffs, np.array(list(map(lambda _: np.prod(np.array(vals).__getitem__(_)), self.terms))))

    def print(self):
        """TODO"""
        raise NotImplementedError


def generate_rel(
    degree: int,
    num_variables: int,
    range_coeffs: List[Tuple[int, int]],
    seed: Optional[int] = None,
    strict_degree=True,
    strict_variables=True,
) -> PolyEq:
    """Sample the coefficients (A, B) of the polynomial equation y = A x + B

    Args:
        strict_degree (bool): if True, require `rel` to have degree strictly equal to `degree`
        strict_variables (bool): if True, require `rel` to use exactly `num_variables`
    Returns:
        `rel` (PolyEq)
    """
    MAX_ATTEMPTS = 100
    if seed is not None:
        random.seed(seed)
    count = 0
    terms = generate_terms(degree, num_variables)
    while count < MAX_ATTEMPTS:
        done = True
        coeffs = [random.randint(r[0], r[1]) for r in range_coeffs]
        if strict_degree and not sum(coeffs[: comb(degree + num_variables - 1, num_variables - 1)]):
            done = False
        if strict_variables:
            for idx in range(num_variables):
                vals = np.zeros(num_variables)
                vals[idx] = 1
                res = np.dot(coeffs[:-1], np.array(list(map(lambda _: np.prod(vals.__getitem__(_)), terms[:-1]))))
                if not res:
                    done = False
                    break
        if done:
            break
        count += 1
        if count >= MAX_ATTEMPTS:
            raise ValueError(
                "Failed to sample valid polynomial equation within "
                + f"{MAX_ATTEMPTS} attempts from ranges {str(range_coeffs)}."
            )
    return PolyEq(degree=degree, num_variables=num_variables, coeffs=coeffs)


class NumeracyScenario(Scenario):
    """
    A task that involves inducing a given polynomial equation given a set of function evaluations.

    Example: (x,y,z) tuples for the polynomial equation z = y^2 + 2x + 5y + 1
        2, -4, 1
        3, -5, 7
        1, -3, -3
        -3, 3, 19
        4, -5, 9
        0, -1, -3
        4, -3, 3
        5, 1, 17
        -3, 2, 9
        4, -5, 9
        -2, 3, -> 21
    """

    name = "numeracy"
    description = "polynomial equation induction"
    tags: List[str] = []

    def __init__(
        self,
        num_train_instances: int,
        num_test_instances: int,
        seed: Optional[int] = 1,
        delimiter: str = ", ",
        degree: int = 1,
        num_variables: int = 1,
        range_coeffs: List[Tuple[int, int]] = [
            (1, 5),
            (-5, 5),
        ],  # min and max range from which to sample each coefficient from, from highest to lowest degree, inclusive
        range_vals: List[Tuple[int, int]] = [
            (-100, 100)
        ],  # min and max range from which to sample variable values from, inclusive
    ):
        num_coeffs = comb(num_variables + degree, degree)
        assert len(range_coeffs) == num_coeffs, (
            f"Expected {num_coeffs} ranges to sample coefficients from for polynomial equation of degree "
            + f"{degree}, received {str(range_coeffs)} of length {len(range_coeffs)}."
        )
        assert len(range_vals) == num_variables
        self.seed = seed
        self.degree = degree
        self.num_variables = num_variables
        self.num_coeffs = num_coeffs
        self.range_coeffs = range_coeffs
        self.range_vals = range_vals
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.delimiter = delimiter

        self.rel = generate_rel(
            degree=degree, num_variables=num_variables, range_coeffs=self.range_coeffs, seed=self.seed
        )

    def get_instances(self) -> List[Instance]:
        def generate_datapoint(rel: PolyEq) -> Tuple[List[str], str]:
            vals = [random.randint(r[0], r[1]) for r in self.range_vals]
            y = rel.eval(vals)
            return list(map(str, vals)), str(y)

        def generate_instance(tags: List[str]):
            """Generate a random instance with `tags`."""
            vals, y = generate_datapoint(self.rel)
            input = self.delimiter.join(vals)
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
