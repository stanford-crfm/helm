import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.math_scenario import (
    MATHScenario,
    get_answer,
    is_equiv_chain_of_thought,
)
from helm.benchmark.scenarios.scenario import Input, Output, Reference


@pytest.mark.skip(reason="competition_math has been disabled on Hugging Face Hub due to a DMCA takedown")
@pytest.mark.scenarios
def test_math_scenario_get_instances():
    math_scenario = MATHScenario(subject="number_theory", level="1")
    with TemporaryDirectory() as tmpdir:
        actual_instances = math_scenario.get_instances(tmpdir)
    assert len(actual_instances) == 77
    assert actual_instances[0].input == Input(text="What is the remainder when (99)(101) is divided by 9?")
    assert actual_instances[0].references == [Reference(output=Output(text="0"), tags=["correct"])]
    assert actual_instances[0].split == "train"


def test_is_equiv_chain_of_thought_missing_answer_is_incorrect():
    """A completion with no extractable answer must not score as correct.

    See https://github.com/stanford-crfm/helm/issues/4374
    """
    gold = r"We get $x=5$, so \boxed 5."  # no braces, so no answer can be extracted
    assert get_answer(gold) is None

    assert is_equiv_chain_of_thought("The answer is 7.", gold) == 0.0
    assert is_equiv_chain_of_thought(r"\boxed{7", gold) == 0.0
    assert is_equiv_chain_of_thought(r"\fbox{7}", r"\fbox{5}") == 0.0

    # Controls: an extractable, matching answer still scores as correct.
    assert is_equiv_chain_of_thought(r"so \boxed{5}.", r"so \boxed{5}.") == 1.0
    assert is_equiv_chain_of_thought(r"so \boxed{7}.", r"so \boxed{5}.") == 0.0
