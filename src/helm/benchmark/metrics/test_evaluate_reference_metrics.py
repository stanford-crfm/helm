from helm.benchmark.metrics.evaluate_reference_metrics import (
    exact_match,
    exact_match_indicator,
    final_number_exact_match,
)


def test_exact_match():
    assert exact_match("33", "33") == 1
    assert exact_match("33", "33 ") == 1
    assert exact_match("33", "34") == 0


def test_exact_match_indicator():
    assert exact_match_indicator("33", "33") == 1
    assert exact_match_indicator("33", "stuff 33") == 1
    assert exact_match_indicator("stuff 33", "33") == 1
    assert exact_match_indicator("33", "33 stuff") == 0


def test_final_number_exact_match():
    assert final_number_exact_match("33", "33") == 1
    assert final_number_exact_match("33", "33 eggs.") == 1
    assert final_number_exact_match("33", "-33") == 0
    assert final_number_exact_match("-33", "-33") == 1
    assert final_number_exact_match("The answer is 33", "\\boxed{33}") == 1
    assert final_number_exact_match("The answer is 33", "\\boxed{33} and 34") == 0
    assert final_number_exact_match("34.2", "2") == 0
    assert final_number_exact_match("342", "342.") == 1
    assert final_number_exact_match("3,420", "3420") == 1
