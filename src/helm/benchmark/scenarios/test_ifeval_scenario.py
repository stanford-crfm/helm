import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.ifeval_scenario import IFEvalScenario
from helm.benchmark.scenarios.scenario import Input, TEST_SPLIT


@pytest.mark.scenarios
def test_ifeval_scenario_get_instances():
    ifeval_scenario = IFEvalScenario()
    with TemporaryDirectory() as tmpdir:
        instances = ifeval_scenario.get_instances(tmpdir)
    assert len(instances) == 541
    assert instances[0].input == Input(
        text=(
            "Write a 300+ word summary of the wikipedia page "
            '"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". '
            "Do not use any commas and highlight at least 3 sections that has titles in markdown format, "
            "for example *highlighted section part 1*, *highlighted section part 2*, "
            "*highlighted section part 3*."
        )
    )
    assert instances[0].split == TEST_SPLIT
    assert instances[0].extra_data
    assert instances[0].extra_data["instruction_ids"] == [
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "length_constraints:number_words",
    ]
    kwargs_groups = instances[0].extra_data["instruction_kwargs"]
    assert all(_ is None for _ in kwargs_groups[0].values())
    assert kwargs_groups[1]["num_highlights"] == 3
    assert all(kwargs_groups[1][key] is None for key in kwargs_groups[1] if key != "num_highlights")
    assert kwargs_groups[2]["relation"] == "at least"
    assert kwargs_groups[2]["num_words"] == 300
    assert all(kwargs_groups[2][key] is None for key in kwargs_groups[2] if key not in {"relation", "num_words"})
