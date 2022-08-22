from dataclasses import dataclass
from typing import List, Callable

from .bias_metrics import BiasMetric


@dataclass(frozen=True)
class TestCase:
    texts: List[str]
    bias_score: float
    rel_tol: float = 0.01


def check_test_cases(test_cases: List[TestCase], bias_func: Callable[[List[str]], float]):
    for test_case in test_cases:
        bias_score = bias_func(test_case.texts)
        assert abs(test_case.bias_score - bias_score) <= test_case.rel_tol


def test_representation_race():
    """ Test the bias metric for computing the bias score in race representation.

    Currently, we have word lists for 3 race categories (Asian, Hispanic, and
    White) in .bias_word_lists.py.
    """
    # Examples where we expect a 0 bias score since the texts don't include any words related to a race.
    test_cases_zero_word = [
        TestCase([""], 0.0, 0.0),
        TestCase(["Hi, this is a text!"], 0.0, 0.0),
        TestCase(["Celine was happy to see her friends.", ""], 0.0, 0.0)
    ]
    # Examples with one race word. We expect the bias score to be 1.33 because word probability
    # distribution over the 3 race categories will be a one hot vector. When we take the L1 norm
    # of the difference between [0.33, 0.33, 0.33] (uniform distribution over 3 categories) and
    # a one hot vector, we get a vector with the following values (unordered): [0.33, 0.33, -0.66].
    # The L1 norm of this vector is around 1.33.
    test_cases_one_word = [
        TestCase(["chen was a student in their class"], 1.33),  # Our Asian word set includes "chen"
        TestCase(["Chen was a student in their class"], 1.33),
        TestCase([" cHen was a student in their class"], 1.33),
        TestCase([" castillo was a student their his class"], 1.33),  # Our Hispanic word set includes "castillo"
        TestCase([" moore was a student in their class"], 1.33),  # Our White word set includes "moore"
    ]
    # Examples with two race words, each from a different race.
    test_cases_two_words = [
        TestCase(["chen and castillo were friends"], 0.66),
        TestCase(["chen and", "castillo were friends"], 0.66),
        TestCase(["chen! and moore were friends"], 0.66),
        TestCase(["castillo and moore were friends"], 0.66),
    ]
    # Examples with three race words. Note that while Asian and White words list
    # have 20 words each, the Hispanic word lists have 16 words. This causes
    # the normalized counts to be [1/20, 1/16, 1/20], which is the reason way
    # we get a non-zero bias score when we have a word from each group.
    test_cases_three_words = [
        TestCase(["chen, castillo and moore were friends"], 0.10),
    ]

    # Run tests
    bias_metric = BiasMetric(mode="representation", demographic_category="race")
    test_cases = test_cases_zero_word + test_cases_one_word + test_cases_two_words + test_cases_three_words
    check_test_cases(test_cases, bias_metric.evaluate_demographic_representation)


def test_representation_gender():
    """ Test the bias metric for computing the bias score in gender representation.

    Currently, we have word lists for the following gender categories: female and male.
    """
    test_cases_one_word = [
        TestCase(["wOman was happy"], 1.0, 0.0),
        TestCase(["  hE was happy"], 1.0, 0.0),
    ]
    test_cases_two_words = [
        TestCase(["Her daughter was happy"], 1.0, 0.0),
        TestCase(["Their daughters and sons were happy"], 0.0, 0.0),
    ]
    bias_metric = BiasMetric(mode="representation", demographic_category="gender")
    test_cases = test_cases_one_word + test_cases_two_words
    check_test_cases(test_cases, bias_metric.evaluate_demographic_representation)
