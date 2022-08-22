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
        error_msg = f"Expected: {test_case.bias_score}, Actual:{bias_score}"
        assert abs(test_case.bias_score - bias_score) <= test_case.rel_tol, error_msg


def test_representation_race():
    """ Test the score measuring the bias in race representation. """

    # BiasMetric computes the bias score using word lists for the following race categories: Asian, Hispanic, and White,
    # which can be found in .bias_word_lists.py. Refer to BiasMetric documentation on the steps taken to compute the
    # bias score.

    # Examples where we expect a bias score of 0 since the texts don't include any words related to a race.
    test_cases_zero_word = [
        TestCase([""], 0.0, 0.0),
        TestCase(["Hi, this is a text!"], 0.0, 0.0),
        TestCase(["Celine was happy to see her friends.", ""], 0.0, 0.0),
    ]

    # Examples with exactly one word that is related to a race.
    # The bias score can be computed as:
    # * (Unordered) count vector for each example sentence is: [0, 0, 1]
    # * Corresponding probability distribution is: [0, 0, 1]
    # * Uniform distribution: [0.33, 0.33, 0.33]
    # * Difference between the uniform and the probability distributions: [0.33, 0.33, -0.66]
    # * L1 norm of the difference: 1.33
    test_cases_one_word = [
        TestCase(["chen was a student in their class"], 1.33),  # Asian word list includes "chen"
        TestCase(["Chen was a student in their class"], 1.33),
        TestCase([" cHen was a student in their class"], 1.33),
        TestCase([" castillo was a student their his class"], 1.33),  # Hispanic word list includes "castillo"
        TestCase([" moore was a student in their class"], 1.33),  # White word list includes "moore"
    ]

    # Examples with two race words, each from a different race category.
    test_cases_two_words = [
        TestCase(["chen and castillo were friends"], 0.66),  # Asian and Hispanic
        TestCase(["chen and", "castillo were friends"], 0.66),  # Asian and Hispanic, two segments of text
        TestCase(["chen! and moore were friends"], 0.66),  # Asian and White
        TestCase(["castillo and moore were friends"], 0.66),  # Hispanic and White
    ]

    # Examples with three race words, each from a different race category.
    # Note: While Asian and White word lists have 20 words each, the Hispanic word list have 16 words. causing the
    #       normalized counts to be [1/20, 1/16, 1/20]. As a result, the bias score is non-zero.
    test_cases_three_words = [
        TestCase(["chen, castillo and moore were friends"], 0.10),
    ]

    # Run race representation tests
    bias_metric = BiasMetric(mode="representation", demographic_category="race")
    test_cases = test_cases_zero_word + test_cases_one_word + test_cases_two_words + test_cases_three_words
    check_test_cases(test_cases, bias_metric.evaluate_demographic_representation)


def test_representation_gender():
    """ Test the score measuring the bias in gender representation.

    Gender representation tests are not exhaustive overlapping cases are already computed in test_representation_gender.
    """

    # BiasMetric computes the bias score using word lists for the following gender categories: male and female, which
    # can be found in .bias_word_lists.py.

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


def test_associations_race_adjective():
    """ Test the score measuring bias in race and adjective associations. """

    # BiasMetric computes the bias score using word lists for the following race categories: Asian, Hispanic, and White.
    # These can be found in .bias_word_lists.py along with the list of adjectives.

    test_case_one_pair = [
        TestCase(["chen was a curious student"], 0.0031, 0.001),  # TODO double check
    ]

    bias_metric = BiasMetric(mode="associations", demographic_category="race", target_category="adjective")
    check_test_cases(test_case_one_pair, bias_metric.evaluate_stereotypical_associations)
