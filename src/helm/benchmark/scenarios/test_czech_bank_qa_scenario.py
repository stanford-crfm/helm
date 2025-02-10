import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.czech_bank_qa_scenario import CzechBankQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input


@pytest.mark.scenarios
def test_czech_bank_qa_scenario_get_instances():
    scenario = CzechBankQAScenario(config_name="default")
    with TemporaryDirectory() as tmpdir:
        actual_instances = scenario.get_instances(tmpdir)
    assert len(actual_instances) == 30
    assert actual_instances[0].input == Input(text="Get the total number of accounts in the system")
    assert len(actual_instances[0].references) == 1
    assert actual_instances[0].references[0].tags == [CORRECT_TAG]
    assert actual_instances[0].references[0].output.text == "SELECT COUNT(*) FROM ACCOUNT"
    assert actual_instances[0].split == "test"
