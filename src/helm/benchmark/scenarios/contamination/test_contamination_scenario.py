import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.contamination.contamination_scenario import ContaminationScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT


@pytest.mark.scenarios
def test_contamination_ts_guessing_multichoice_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the Contamination scenario using the TS-Guessing MultiChoice strategy
        scenario = ContaminationScenario(
            dataset="bluex",
            strategy="ts_guessing_question_multichoice",
            language="pt"
        )
        instances = scenario.get_instances(tmpdir)

        assert len(instances) > 0
        assert instances[0].split == TEST_SPLIT
        assert "[MASK]" in instances[0].input.text
        assert len(instances[0].references) > 0


@pytest.mark.scenarios
def test_contamination_ts_guessing_base_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the Contamination scenario using the TS-Guessing Base strategy
        scenario = ContaminationScenario(
            dataset="bluex",
            strategy="ts_guessing_question_base",
            language="pt"
        )
        instances = scenario.get_instances(tmpdir)

        assert len(instances) > 0
        assert instances[0].split == TEST_SPLIT
        assert "[MASK]" in instances[0].input.text
        assert len(instances[0].references) > 0


@pytest.mark.scenarios
def test_contamination_invalid_strategy():
    with TemporaryDirectory() as tmpdir:
        # Test that an invalid strategy raises the appropriate error
        scenario = ContaminationScenario(
            dataset="bluex",
            strategy="invalid_strategy",
            language="pt"
        )
        
        with pytest.raises(ValueError):
            scenario.get_instances(tmpdir)