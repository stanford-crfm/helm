# Test metrics
from typing import List

import numpy as np
import pytest
from helm.benchmark.metrics.disinformation_metrics import _monte_carlo_entropy, _self_bleu
from helm.common.request import GeneratedOutput, Token

# Test tokens
_TEST_1_TOKENS: List[Token] = [
    Token("This", logprob=-0.25),
    Token("is", logprob=-0.25),
    Token("a", logprob=-0.25),
    Token("test", logprob=-0.25),
]
_TEST_2_TOKENS: List[Token] = [
    Token("This", logprob=-0.25),
    Token("is", logprob=-0.25),
    Token("another", logprob=-0.5),
    Token("test", logprob=-0.25),
]
_TEST_EMPTY_TOKENS: List[Token] = []
test_empty_str_tokens: List[Token] = [
    Token("", logprob=0),
]

# Test Sequences (two standard, one with an empty token, and one with no tokens)
_TEST_1 = GeneratedOutput(text="This is a test", logprob=-1, tokens=_TEST_1_TOKENS)
_TEST_2 = GeneratedOutput(text="This is another test", logprob=-1.25, tokens=_TEST_2_TOKENS)
_TEST_EMPTY = GeneratedOutput(text="", logprob=-float("nan"), tokens=_TEST_EMPTY_TOKENS)
_TEST_EMPTY_STR = GeneratedOutput(text="", logprob=0, tokens=test_empty_str_tokens)


# Test Self-BLEU
def test_self_bleu_with_self():
    score = _self_bleu([_TEST_1, _TEST_1])
    assert score == pytest.approx(100)


def test_self_blue_with_other():
    score = _self_bleu([_TEST_1, _TEST_2])
    assert 0 < score < 100


def test_self_blue_one_sequence():
    score = _self_bleu([_TEST_1])
    assert score == 0


def test_self_blue_one_full_one_empty():
    score = _self_bleu([_TEST_1, _TEST_EMPTY_STR])
    assert score == 0


# Test MC Entropy
def test_mc_entropy_with_self():
    score = _monte_carlo_entropy([_TEST_1, _TEST_1])
    assert score == pytest.approx(-_TEST_1.logprob)


def test_mc_entropy_with_other():
    score = _monte_carlo_entropy([_TEST_1, _TEST_2])
    assert score == pytest.approx(-(_TEST_1.logprob + _TEST_2.logprob) / 2)


def test_mc_entropy_one_sequence():
    score = _monte_carlo_entropy([_TEST_1])
    assert score == -_TEST_1.logprob


def test_mc_entropy_one_full_one_empty():
    score = _monte_carlo_entropy([_TEST_EMPTY_STR])
    assert score == _TEST_EMPTY_STR.logprob


def test_mc_entropy_with_no_tokens():
    score = _monte_carlo_entropy([_TEST_EMPTY])
    assert np.isnan(score)
