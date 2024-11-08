import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.mmlu_scenario import MMLUScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_mmlu_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = MMLUScenario(subject="abstract_algebra")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 116
        assert instances[0].input == Input(text="Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.")
        assert instances[0].references == [
            Reference(output=Output(text="0"), tags=[]),
            Reference(output=Output(text="1"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="2"), tags=[]),
            Reference(output=Output(text="3"), tags=[]),
        ]
        assert instances[0].split == "train"

        scenario = MMLUScenario(subject="anatomy")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 154
        assert instances[0].input == Input(text="What is the embryological origin of the hyoid bone?")
        assert instances[0].references == [
            Reference(output=Output(text="The first pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The first and second pharyngeal arches"), tags=[]),
            Reference(output=Output(text="The second pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The second and third pharyngeal arches"), tags=[CORRECT_TAG]),
        ]
        assert instances[0].split == "train"
