import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.med_qa_scenario import MedQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_med_qa_scenario():
    scenario = MedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 12723
    assert instances[0].input == Input(
        text=(
            "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it"
            " started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She"
            " otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C),"
            " blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on"
            " room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus."
            " Which of the following is the best treatment for this patient?"
        )
    )
    assert instances[0].references == [
        Reference(output=Output(text="Ampicillin"), tags=[]),
        Reference(output=Output(text="Ceftriaxone"), tags=[]),
        Reference(output=Output(text="Doxycycline"), tags=[]),
        Reference(output=Output(text="Nitrofurantoin"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == "train"
