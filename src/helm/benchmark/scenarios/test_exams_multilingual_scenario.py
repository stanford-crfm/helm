import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.exams_multilingual_scenario import EXAMSMultilingualScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TRAIN_SPLIT, Input


@pytest.mark.scenarios
def test_exam_multilingual_scenario_get_instances():
    scenario = EXAMSMultilingualScenario(language="Bulgarian", subject="Physics")
    with TemporaryDirectory() as tmpdir:
        actual_instances = scenario.get_instances(tmpdir)
    assert len(actual_instances) == 393
    assert actual_instances[0].id == "4c05bbb8-7729-11ea-9116-54bef70b159e"
    assert actual_instances[0].input == Input(text="Наелектризирането по индукция се обяснява с: ")
    assert len(actual_instances[0].references) == 4
    assert actual_instances[0].references[0].output.text == "преразпределение на положителните йони в тялото"
    assert actual_instances[0].references[0].tags == []
    assert (
        actual_instances[0].references[1].output.text == "предаване на електрони от неутрално на наелектризирано тяло"
    )
    assert actual_instances[0].references[1].tags == []
    assert (
        actual_instances[0].references[2].output.text == "предаване на електрони от наелектризирано на неутрално тяло"
    )
    assert actual_instances[0].references[2].tags == []
    assert actual_instances[0].references[3].output.text == "преразпределение на свободните електрони в тялото"
    assert actual_instances[0].references[3].tags == [CORRECT_TAG]
    assert actual_instances[0].split == TRAIN_SPLIT
