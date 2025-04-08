import pytest
import re
from tempfile import TemporaryDirectory
from helm.benchmark.scenarios.infinite_bench_sum_scenario import InfiniteBenchSumScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


def count_words(text: str) -> int:
    return len(re.split(r"\s+", text.strip()))


@pytest.mark.scenarios
def test_infinite_bench_sum_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = InfiniteBenchSumScenario(min_num_words=0, max_num_words=10000000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 103
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 1745528
        assert instances[0].extra_data
        assert instances[0].extra_data["word_count"] == 308762
        references = instances[0].references
        assert len(references[0].output.text) == 2865
        assert references[0].tags == [CORRECT_TAG]

        scenario = InfiniteBenchSumScenario(min_num_words=0, max_num_words=100000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 48
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 381778
        assert instances[0].extra_data
        assert instances[0].extra_data["word_count"] == 69458
        references = instances[0].references
        assert len(references[0].output.text) == 4217
        assert references[0].tags == [CORRECT_TAG]

        scenario = InfiniteBenchSumScenario(min_num_words=30000, max_num_words=80000)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 32
        assert instances[0].split == "test"
        assert len(instances[1].input.text) == 383396
        assert instances[1].extra_data
        assert instances[1].extra_data["word_count"] == 68482
        references = instances[1].references
        assert len(references[0].output.text) == 5667
        assert references[0].tags == [CORRECT_TAG]
