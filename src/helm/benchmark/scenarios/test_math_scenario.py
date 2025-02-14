import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.math_scenario import MATHScenario
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
