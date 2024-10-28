import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.mmlu_pro import MMLUProScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_mmlu_pro_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the "abstract_algebra" subject
        scenario = MMLUProScenario(subject="math")
        instances = scenario.get_instances(tmpdir)
        #assert len(instances) == 116
        assert instances[1].input == Input(text="Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?")

        # Ensure it handles up to 10 answer options
        assert instances[1].references == [
            Reference(output=Output(text="ST + TS is the identity map of V onto itself."), tags=[]),
            Reference(output=Output(text="TS = 0"), tags=[]),
            Reference(output=Output(text="ST = 1"), tags=[]),
            Reference(output=Output(text="ST - TS = 0"), tags=[]),
            Reference(output=Output(text="ST = T"), tags=[]),
            Reference(output=Output(text="ST = 0"), tags=[]),
            Reference(output=Output(text="ST = TS"), tags=[]),
            Reference(output=Output(text="ST - TS is the identity map of V onto itself."), tags=[CORRECT_TAG]),
            Reference(output=Output(text="TS = T"), tags=[]),
            Reference(output=Output(text="ST = S"), tags=[]),
        ]
        assert instances[1].split == "train"

        # Optional: check if the explanation is properly included (if provided in the dataset)
        #assert hasattr(instances[0], "explanation")

        # Test for the "anatomy" subject
        scenario = MMLUProScenario(subject="health")
        instances = scenario.get_instances(tmpdir)
       # assert len(instances) == 154
        assert instances[0].input == Input(text="Which of the following is the body cavity that contains the pituitary gland?")

        # Check references with more answer choices and correct tagging
        assert instances[0].references == [
            Reference(output=Output(text="Ventral"), tags=[]),
            Reference(output=Output(text="Dorsal"), tags=[]),
            Reference(output=Output(text="Buccal"), tags=[]),
            Reference(output=Output(text="Thoracic"), tags=[]),
            Reference(output=Output(text="Pericardial"), tags=[]),
            Reference(output=Output(text="Abdominal"), tags=[]),
            Reference(output=Output(text="Spinal"), tags=[]),
            Reference(output=Output(text="Pelvic"), tags=[]),
            Reference(output=Output(text="Pleural"), tags=[]),
            Reference(output=Output(text="Cranial"), tags=[CORRECT_TAG]),
        ]
        assert instances[0].split == "train"

        # Again, check for the presence of an explanation
       # assert hasattr(instances[0], "explanation")
