import pytest
from tempfile import TemporaryDirectory

from datasets.exceptions import DatasetNotFoundError

from helm.benchmark.scenarios.ewok_scenario import EWoKScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


@pytest.mark.scenarios
def test_ewok_scenario():
    scenario = EWoKScenario()
    with TemporaryDirectory() as tmpdir:
        try:
            instances = scenario.get_instances(tmpdir)
        except (DatasetNotFoundError, Exception):
            # Catch generic Exception here because the Hugging Face Hub client can raise
            # an authentication issue as the following exception:
            # huggingface_hub.utils._errors.GatedRepoError: 401 Client Error.
            # but GatedRepoError is a type in a private package, so we avoid referencing it
            pytest.skip("Unable to access gated dataset on Hugging Face Hub; skipping test")
    assert len(instances) == 8750
    assert "believes" in instances[0].input.text
    assert len(instances[0].references) == 2
    assert "inside" in instances[0].references[0].output.text
    assert instances[0].references[0].tags == [CORRECT_TAG]
    assert "outside" in instances[0].references[1].output.text
    assert instances[0].references[1].tags == []
    assert instances[0].split == "test"
