import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.gsm_scenario import GSM8KScenario
from helm.benchmark.scenarios.scenario import Input, Output, Reference


@pytest.mark.scenarios
def test_gsm_scenario_get_instances():
    math_scenario = GSM8KScenario()
    with TemporaryDirectory() as tmpdir:
        actual_instances = math_scenario.get_instances(tmpdir)
    assert len(actual_instances) == 8792
    assert actual_instances[0].input == Input(
        text=(
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many"
            " clips did Natalia sell altogether in April and May?"
        )
    )
    assert actual_instances[0].references == [
        Reference(
            output=Output(
                text=(
                    "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips"
                    " altogether in April and May. The answer is 72."
                )
            ),
            tags=["correct"],
        )
    ]
    assert actual_instances[0].split == "train"
