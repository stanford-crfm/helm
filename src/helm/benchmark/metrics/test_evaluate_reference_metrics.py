import pytest
from helm.benchmark.metrics.evaluate_reference_metrics import (
    bleu_1,
    chinese_bleu_1,
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
    # Commas in prose after the final number must not be extracted as the "final number" (#4337)
    assert final_number_exact_match("The answer is 72.", "The answer is 72, in total, so the farmer earns that.") == 1
    assert final_number_exact_match("The answer is 72.", "The answer is 72, which is the total.") == 1
    assert final_number_exact_match("3", "The answer is 3, yes, really.") == 1
    # A comma-only run in the gold must not be extracted either
    assert final_number_exact_match("72, in total,", "72") == 1
    # The decimal point is escaped, so an arbitrary separator no longer glues digits into one number
    assert final_number_exact_match("4", "3 4") == 1
    assert final_number_exact_match("5", "1/2.5 hours") == 0


def test_bleu_1():
    assert bleu_1("the quick brown fox jumped over the lazy dog", "the fox jumped over the dog") == pytest.approx(
        0.6065306597126334
    )


def test_chinese_bleu_1():
    assert chinese_bleu_1(
        "太祖武皇帝，沛國譙人也，姓曹，諱操，字孟德，漢相國參之後。", "太祖武皇帝，沛國譙人也，漢相國參之後。"
    ) == pytest.approx(0.5907775139012316)
