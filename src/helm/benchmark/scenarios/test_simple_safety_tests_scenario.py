import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.simple_safety_tests_scenario import SimpleSafetyTestsScenario
from helm.benchmark.scenarios.scenario import Input


@pytest.mark.scenarios
def test_simple_safety_tests_scenario_get_instances():
    sst_scenario = SimpleSafetyTestsScenario()
    with TemporaryDirectory() as tmpdir:
        actual_instances = sst_scenario.get_instances(tmpdir)
    assert len(actual_instances) == 100
