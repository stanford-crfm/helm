import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.ewok_scenario import EWoKScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_legalbench_scenario():
    scenario = EWoKScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 8748
    assert instances[0].input == Input(text="Ali believes that the candle is in the bakery.")
    assert instances[0].references == [
        Reference(output=Output(text="Ali is in the bakery. Ali sees the candle inside."), tags=[CORRECT_TAG]),
        Reference(output=Output(text="Ali is in the bakery. Ali sees the candle outside."), tags=[]),
    ]
    assert instances[0].split == "test"
