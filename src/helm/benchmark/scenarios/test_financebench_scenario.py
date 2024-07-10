import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.financebench_scenario import FinanceBenchScenario
from helm.benchmark.scenarios.scenario import TRAIN_SPLIT, Input


# @pytest.mark.scenarios
def test_air_2024_scenario_get_instances():
    scenario = FinanceBenchScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 150
    assert (
        instances[0].input.text
        == "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
    )
    assert len([instance for instance in instances if instance.split == TRAIN_SPLIT]) == 10
    assert len(instances[0].references) == 1
    assert instances[0].references[0].output.text == "$1577.00"
    assert instances[0].split == "test"
