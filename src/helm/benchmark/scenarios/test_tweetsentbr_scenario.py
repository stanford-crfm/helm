import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.tweetsentbr_scenario import TweetSentBRScenario
from helm.benchmark.scenarios.scenario import TRAIN_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_tweetsentbr_scenario():
    tweetsentbr = TweetSentBRScenario()
    with TemporaryDirectory() as tmpdir:
        instances = tweetsentbr.get_instances(tmpdir)
    assert len(instances) == 2085
    assert instances[0].split == TRAIN_SPLIT

    assert instances[0].input.text.startswith("joca tá com a corda toda 😂 😂 😂 😂")
    assert len(instances[0].input.text) == 32

    assert instances[0].references == [
        Reference(
            output=Output(text="Positivo"),
            tags=[CORRECT_TAG],
        )
    ]
