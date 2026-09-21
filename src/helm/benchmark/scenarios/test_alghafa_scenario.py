import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.alghafa_scenario import AlGhafaScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input


@pytest.mark.scenarios
def test_alghafa_scenario_get_instances():
    scenario = AlGhafaScenario(subset="mcq_exams_test_ar")
    with TemporaryDirectory() as tmpdir:
        actual_instances = scenario.get_instances(tmpdir)
    assert len(actual_instances) == 562
    assert actual_instances[0].id == "id0_test"
    assert actual_instances[0].input == Input(
        text=(
            'قال علي بن أبي طالب رضي الله عنه عن عمر بن الخطاب رضي الله عنه " إن كنا لنرى إن في القرآن كلاماً من كلامه ورأياً من رأيه " دلت هذه العبارة على سمة وصفة من صفات عمر بن الخطاب رضي الله عنه هي'  # noqa: E501
        )
    )
    assert len(actual_instances[0].references) == 4
    assert actual_instances[0].references[0].output.text == "الشجاعة"
    assert actual_instances[0].references[0].tags == []
    assert actual_instances[0].references[1].output.text == "نزل القرآن الكريم موافقاً لرأيه في عدة مواضع"
    assert actual_instances[0].references[1].tags == [CORRECT_TAG]
    assert actual_instances[0].references[2].output.text == "الشدة في الحق مع اللين والرحمة ."
    assert actual_instances[0].references[2].tags == []
    assert actual_instances[0].references[3].output.text == "التواضع"
    assert actual_instances[0].references[3].tags == []
    assert actual_instances[0].split == "test"
