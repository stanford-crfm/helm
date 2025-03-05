import pytest
from tempfile import TemporaryDirectory
from helm.benchmark.scenarios.infinite_bench_sum_scenario import InfiniteBenchSumScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG


@pytest.mark.scenarios
def test_infinite_bench_sum_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = InfiniteBenchSumScenario()
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 103
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 1745528
        assert len(instances[0].input.text.split(' ')) == 317253
        references = instances[0].references
        assert len(references[0].output.text) == 2865
        assert references[0].tags == [CORRECT_TAG]

        scenario = InfiniteBenchSumScenario(word_lower_bound=0, word_upper_bound=100e3)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 48
        assert instances[0].split == "test"
        assert len(instances[0].input.text) == 381778
        assert len(instances[0].input.text.split(' ')) == 68651
        references = instances[0].references
        assert len(references[0].output.text) == 4217
        assert references[0].tags == [CORRECT_TAG]

        scenario = InfiniteBenchSumScenario(word_lower_bound=30e3, word_upper_bound=80e3)
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 32
        assert instances[0].split == "test"
        assert len(instances[1].input.text) == 383396
        assert len(instances[1].input.text.split(' ')) == 63542
        references = instances[1].references
        assert len(references[0].output.text) == 5667
        assert references[0].tags == [CORRECT_TAG]
