import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.wildbench_scenario import WildBenchScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT


@pytest.mark.scenarios
def test_wildbench_scenario_get_instances():
    wildbench_scenario = WildBenchScenario(subset="v2")
    with TemporaryDirectory() as tmpdir:
        instances = wildbench_scenario.get_instances(tmpdir)
    assert len(instances) == 1024
    assert instances[0].split == TEST_SPLIT
    assert instances[0].extra_data
