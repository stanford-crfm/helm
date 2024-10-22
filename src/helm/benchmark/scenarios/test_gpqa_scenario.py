import pytest
from tempfile import TemporaryDirectory
from helm.benchmark.scenarios.gpqa_scenario import GPQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


@pytest.mark.scenarios
def test_gpqa_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = GPQAScenario(subset="gpqa_main")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 448
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 689
        references = instances[0].references
        assert len(references[0].output.text) == 10
        assert len(references[1].output.text) == 6
        assert len(references[2].output.text) == 9
        assert len(references[3].output.text) == 7
        assert references[3].tags == [CORRECT_TAG]

        scenario = GPQAScenario(subset="gpqa_diamond")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 198
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 262
        references = instances[0].references
        assert len(references[0].output.text) == 8
        assert len(references[1].output.text) == 9
        assert len(references[2].output.text) == 8
        assert len(references[3].output.text) == 8
        assert references[3].tags == [CORRECT_TAG]

        scenario = GPQAScenario(subset="gpqa_extended")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 546
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 689
        references = instances[0].references
        assert len(references[0].output.text) == 10
        assert len(references[1].output.text) == 6
        assert len(references[2].output.text) == 9
        assert len(references[3].output.text) == 7
        assert references[3].tags == [CORRECT_TAG]
