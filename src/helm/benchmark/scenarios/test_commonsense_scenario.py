import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.commonsense_scenario import OpenBookQA
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_openbookqa_scenario():
    scenario = OpenBookQA()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 5457
    assert instances[0].input == Input(text="The sun is responsible for")
    assert instances[0].references == [
        Reference(output=Output(text="puppies learning new tricks"), tags=[]),
        Reference(output=Output(text="children growing up and getting old"), tags=[]),
        Reference(output=Output(text="flowers wilting in a vase"), tags=[]),
        Reference(output=Output(text="plants sprouting, blooming and wilting"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == "train"
