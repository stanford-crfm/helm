import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.imdb_ptbr_scenario import IMDB_PTBRScenario
from helm.benchmark.scenarios.scenario import TRAIN_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_imdb_ptbr_scenario():
    imdb_ptbr = IMDB_PTBRScenario()
    with TemporaryDirectory() as tmpdir:
        instances = imdb_ptbr.get_instances(tmpdir)
    assert len(instances) == 30000
    assert instances[0].split == TRAIN_SPLIT

    assert instances[10].input.text.startswith(
        "Foi ótimo ver algumas das minhas estrelas favoritas de 30 anos atrás, "
        "incluindo John Ritter, Ben Gazarra e Audrey Hepburn."
    )
    assert len(instances[10].input.text) == 1549

    assert instances[10].references == [
        Reference(
            output=Output(text="negativo"),
            tags=[CORRECT_TAG],
        )
    ]
