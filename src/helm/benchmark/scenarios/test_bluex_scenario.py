import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.bluex_scenario import BLUEXScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_bluex_scenario():
    scenario = BLUEXScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) > 100

    assert instances[100].split == TEST_SPLIT

    assert instances[0].input.text.startswith("Rubião fitava a enseada, - eram oito horas da manhã Quem o visse")

    assert len(instances[0].input.text) == 1011

    assert instances[0].references == [
        Reference(
            output=Output(
                text='a contemplação das paisagens naturais, como se lê em "ele admirava aquele pedaço de água quieta".'
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text='a presença de um narrador-personagem, como se lê em "em verdade vos digo que pensava em '
                'outra coisa".'
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text='a sobriedade do protagonista ao avaliar o seu percurso, como se lê em "Cotejava o passado com '
                "o presente."
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text='o sentido místico e fatalista que rege os destinos, como se lê em "Deus escreve direito por '
                'linhas tortas".'
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text='a reversibilidade entre o cômico e o trágico, como se lê em "de modo que o que parecia uma '
                'desgraça...".'
            ),
            tags=[CORRECT_TAG],
        ),
    ]

    assert instances[0].references[4].is_correct
