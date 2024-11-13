import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.wildbench_scenario import WildBenchScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT


@pytest.mark.scenarios
def test_wildbench_scenario_get_instances():
    wildbench_scenario = WildBenchScenario(subset="v2")
    with TemporaryDirectory() as tmpdir:
        instances = wildbench_scenario.get_instances(tmpdir)
    assert len(instances) == 1024
    assert instances[0].split == TEST_SPLIT
    assert instances[0].extra_data

    assert instances[0].extra_data["user_query"].startswith("add 10 more balanced governments[aoc2]\n{\n\tGovernment")
    assert len(instances[0].extra_data["user_query"]) == 17619

    assert instances[0].extra_data["baseline_outputs"]
    assert instances[0].extra_data["baseline_outputs"]["gpt-4-turbo-2024-04-09"].startswith("Here are 10 addition")
    assert len(instances[0].extra_data["baseline_outputs"]["gpt-4-turbo-2024-04-09"]) == 10574
    assert instances[0].extra_data["baseline_outputs"]["claude-3-haiku-20240307"].startswith("Here are 10 more bal")
    assert len(instances[0].extra_data["baseline_outputs"]["claude-3-haiku-20240307"]) == 7873
    assert instances[0].extra_data["baseline_outputs"]["Llama-2-70b-chat-hf"] == ""
