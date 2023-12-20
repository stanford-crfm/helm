import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.math_scenario import MATHScenario
from helm.benchmark.scenarios.scenario import Input, Output, Reference


# TODO: Fix the test for newer versions of diffusers: https://github.com/stanford-crfm/helm/issues/2168
@pytest.mark.skip(
    reason="Incompatible with newer versions with diffusers>0.24.0. Fails with "
    '"Loading a dataset cached in a LocalFileSystem is not supported"'
)
def test_math_scenario_get_instances():
    math_scenario = MATHScenario(subject="number_theory", level="1")
    with TemporaryDirectory() as tmpdir:
        actual_instances = math_scenario.get_instances(tmpdir)
    assert len(actual_instances) == 77
    assert actual_instances[0].input == Input(text="What is the remainder when (99)(101) is divided by 9?")
    assert actual_instances[0].references == [
        Reference(output=Output(text="0", multimedia_content=None), tags=["correct"])
    ]
    assert actual_instances[0].split == "train"
