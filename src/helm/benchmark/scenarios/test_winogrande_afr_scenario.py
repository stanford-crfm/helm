import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.winogrande_afr_scenario import Winogrande_Afr_Scenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_winogrande_afr_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = Winogrande_Afr_Scenario(lang="am")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 3674
        assert instances[0].input == Input(text="ሳራ ከማሪያ በጣም የተሻለች የቀዶ ጥገና ሐኪም ስለነበረች ሁልጊዜ _ ቀላል ህመሞችን ታክማለች.")
        assert instances[0].references == [
            Reference(output=Output(text="ሳራ"), tags=[]),
            Reference(output=Output(text="ማሪያ"), tags=[CORRECT_TAG]),
        ]
        assert instances[0].split == "train"
