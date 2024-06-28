import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.legalbench_scenario import LegalBenchScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_legalbench_scenario():
    scenario = LegalBenchScenario(subset="abercrombie")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 100
    assert instances[0].input == Input(text='Description: The mark "Ivory" for a product made of elephant tusks.')
    assert instances[0].references == [
        Reference(output=Output(text="generic"), tags=["correct"]),
    ]
    assert instances[0].split == "train"

    scenario = LegalBenchScenario(subset="proa")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 100
    assert instances[0].input == Input(
        text="Statute: Amendments to pleadings must be filed within 15 days of the filing of the initial pleading."
    )
    assert instances[0].references == [
        Reference(output=Output(text="No"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == "train"
