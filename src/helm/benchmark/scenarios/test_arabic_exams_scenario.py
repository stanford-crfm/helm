import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.arabic_exams_scenario import ArabicEXAMSScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input


@pytest.mark.scenarios
def test_arabic_exams_get_instances():
    scenario = ArabicEXAMSScenario(subject="all")
    with TemporaryDirectory() as tmpdir:
        actual_instances = scenario.get_instances(tmpdir)
    assert len(actual_instances) == 562
    assert actual_instances[0].id == "Islamic Studies-0"
    assert actual_instances[0].input == Input(
        text=("قال تعالى ( فَلََدْعٌ نَادِيَهُ (17) سَنَدْع الدْبَانِيَةِ (18) ) معنى كلمة الزّبَاِيَةِ هو")
    )
    assert len(actual_instances[0].references) == 4
    assert actual_instances[0].references[2].output.text == "خزنة جهنم"
    assert actual_instances[0].references[2].tags == [CORRECT_TAG]
    assert actual_instances[0].split == "test"
