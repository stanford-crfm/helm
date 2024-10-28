import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.mmlu_pro import MMLUProScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_mmlu_pro_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the "abstract_algebra" subject
        scenario = MMLUProScenario(subject="abstract_algebra")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 116
        assert instances[0].input == Input(text="Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.")

        # Ensure it handles up to 10 answer options
        assert instances[0].references == [
            Reference(output=Output(text="0"), tags=[]),
            Reference(output=Output(text="1"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="2"), tags=[]),
            Reference(output=Output(text="3"), tags=[]),
            Reference(output=Output(text="4"), tags=[]),
            Reference(output=Output(text="5"), tags=[]),
            Reference(output=Output(text="6"), tags=[]),
            Reference(output=Output(text="7"), tags=[]),
            Reference(output=Output(text="8"), tags=[]),
            Reference(output=Output(text="9"), tags=[]),
        ]
        assert instances[0].split == "train"

        # Optional: check if the explanation is properly included (if provided in the dataset)
        assert hasattr(instances[0], "explanation")

        # Test for the "anatomy" subject
        scenario = MMLUProScenario(subject="anatomy")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 154
        assert instances[0].input == Input(text="What is the embryological origin of the hyoid bone?")

        # Check references with more answer choices and correct tagging
        assert instances[0].references == [
            Reference(output=Output(text="The first pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The first and second pharyngeal arches"), tags=[]),
            Reference(output=Output(text="The second pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The second and third pharyngeal arches"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="The fourth pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The fifth pharyngeal arch"), tags=[]),
            Reference(output=Output(text="The sixth pharyngeal arch"), tags=[]),
            Reference(output=Output(text="None of the above"), tags=[]),
            Reference(output=Output(text="All of the above"), tags=[]),
            Reference(output=Output(text="Other"), tags=[]),
        ]
        assert instances[0].split == "train"

        # Again, check for the presence of an explanation
        assert hasattr(instances[0], "explanation")
