import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.bigcodebench_scenario import BigCodeBenchScenario
from helm.benchmark.scenarios.scenario import Input, TEST_SPLIT


@pytest.mark.scenarios
def test_bigcodebench_scenario_get_instances():
    bigcodebench_scenario = BigCodeBenchScenario("v0.1.2")
    with TemporaryDirectory() as tmpdir:
        instances = bigcodebench_scenario.get_instances(tmpdir)
    assert len(instances) == 1140
    assert instances[0].id == "BigCodeBench/0"
    assert instances[0].input == Input(
        text=(
            "Calculates the average of the sums of absolute differences between each pair "
            "of consecutive numbers for all permutations of a given list. Each permutation "
            "is shuffled before calculating the differences. Args: - numbers (list): A list "
            "of numbers. Default is numbers from 1 to 10.\nThe function should output with:\n"
            "    float: The average of the sums of absolute differences for each shuffled permutation "
            "of the list.\nYou should write self-contained code starting with:\n```\nimport itertools\n"
            "from random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n```"
        )
    )
    assert instances[0].split == TEST_SPLIT
