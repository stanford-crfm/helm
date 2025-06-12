import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.infinite_bench_en_qa_scenario import InfiniteBenchEnQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


@pytest.mark.scenarios
def test_infinite_bench_en_qa_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = InfiniteBenchEnQAScenario(max_num_words=10000000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 351
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 381829
        assert len(instances[0].references) == 1
        assert len(instances[0].references[0].output.text) == 8
        assert instances[0].references[0].tags == [CORRECT_TAG]
