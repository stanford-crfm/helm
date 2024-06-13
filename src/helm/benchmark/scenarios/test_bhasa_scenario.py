import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.bhasa_scenario import *
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Scenario, Input, Output, PassageQuestionInput, Reference


@pytest.mark.scenarios
def test_indicqa_qa_ta_scenario():
    scenario = IndicQA_QA_TA_Scenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 90
    assert instances[0].input == Input(text="Is 10 even or odd?")
    assert instances[0].references == [
        Reference(output=Output(text="Even"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="Odd"), tags=[]),
    ]
    assert instances[0].split == "train"


@pytest.mark.scenarios
def test_simple_short_answer_qa_scenario():
    scenario = SimpleShortAnswerQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 90
    assert instances[0].input == Input(text="Is 10 even or odd?")
    assert instances[0].references == [
        Reference(output=Output(text="Even"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == "train"


@pytest.mark.scenarios
def test_simple_classification_scenario():
    scenario = SimpleClassificationScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 90
    assert instances[0].input == Input(text="10")
    assert instances[0].references == [
        Reference(output=Output(text="Even"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="Odd"), tags=[]),
    ]
    assert instances[0].split == "train"