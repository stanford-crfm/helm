import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.mmlu_clinical_afr_scenario import MMLU_Clinical_Afr_Scenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_mmlu_clinical_afr_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = MMLU_Clinical_Afr_Scenario(subject="clinical_knowledge", lang="am")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 299
        assert instances[0].input == Input(text="ለሁሉም የጡንቻ መኮማተር ዓይነቶች የሚያስፈልገው ኢኔርጅ የሚቀርበው ከሚከተሉት ነው፦")
        assert instances[0].references == [
            Reference(output=Output(text="ATP።"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="ADP።"), tags=[]),
            Reference(output=Output(text="ፎስፎክሬቲን።"), tags=[]),
            Reference(output=Output(text="ኦክስዳቲቪ ፎስፎሪሌሽን።"), tags=[]),
        ]
        assert instances[0].split == "train"
