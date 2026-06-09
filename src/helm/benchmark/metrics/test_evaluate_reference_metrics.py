import pytest
from helm.benchmark.metrics.evaluate_reference_metrics import (
    bleu_1,
    chinese_bleu_1,
    exact_match,
    exact_match_indicator,
    f1_score,
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


def test_bleu_1():
    assert bleu_1("the quick brown fox jumped over the lazy dog", "the fox jumped over the dog") == pytest.approx(
        0.6065306597126334
    )


def test_chinese_bleu_1():
    assert chinese_bleu_1(
        "太祖武皇帝，沛國譙人也，姓曹，諱操，字孟德，漢相國參之後。", "太祖武皇帝，沛國譙人也，漢相國參之後。"
    ) == pytest.approx(0.5907775139012316)


def test_f1_score_normal():
    # Standard QA: partial overlap
    assert f1_score("the cat sat on the mat", "the cat sat") == pytest.approx(2 / 3)
    # Identical strings
    assert f1_score("quick brown fox", "quick brown fox") == pytest.approx(1.0)
    # No overlap
    assert f1_score("cat", "dog") == pytest.approx(0.0)


def test_f1_score_article_gold():
    # Regression for issue #2298: gold is a single English article letter.
    # Before the fix, normalize_text("A") → "" and f1_score returned 0.0 even
    # when the prediction was also "A".
    assert f1_score("A", "A") == pytest.approx(1.0)
    assert f1_score("A", "B") == pytest.approx(0.0)
    assert f1_score("An", "An") == pytest.approx(1.0)
    assert f1_score("The", "The") == pytest.approx(1.0)
    # Pred is the article, gold is not
    assert f1_score("cat", "a") == pytest.approx(0.0)


def test_f1_score_article_in_longer_string():
    # Article removal in a multi-word string should still apply (existing behavior).
    # "a cat" → "cat" after normalization; "the cat" → "cat"; both match.
    assert f1_score("a cat", "the cat") == pytest.approx(1.0)
    # Extra non-article words lower the score.
    assert f1_score("a cat sat", "a cat") == pytest.approx(2 / 3)


def test_f1_score_empty_pred():
    # Empty prediction always yields 0.
    assert f1_score("cat", "") == pytest.approx(0.0)
    assert f1_score("A", "") == pytest.approx(0.0)
