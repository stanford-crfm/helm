import pytest
import re
from tempfile import TemporaryDirectory
from helm.benchmark.scenarios.infinite_bench_en_sum_scenario import InfiniteBenchEnSumScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


def count_words(text: str) -> int:
    return len(re.split(r"\s+", text.strip()))


@pytest.mark.scenarios
def test_infinite_bench_en_sum_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = InfiniteBenchEnSumScenario(max_num_words=10000000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 103
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 1745528
        references = instances[0].references
        assert len(references[0].output.text) == 2865
        assert references[0].tags == [CORRECT_TAG]

        scenario = InfiniteBenchEnSumScenario(max_num_words=100000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 48
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 381778
        references = instances[0].references
        assert len(references[0].output.text) == 4217
        assert references[0].tags == [CORRECT_TAG]
