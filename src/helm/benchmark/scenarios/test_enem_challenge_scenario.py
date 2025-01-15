import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.enem_challenge_scenario import ENEMChallengeScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_enem_challenge_scenario():
    enem_scenario = ENEMChallengeScenario()
    with TemporaryDirectory() as tmpdir:
        instances = enem_scenario.get_instances(tmpdir)
    assert len(instances) == 1431
    assert instances[0].split == TEST_SPLIT

    assert instances[0].input.text.startswith(
        "A atmosfera terrestre é composta pelos gases nitrogênio (N2) e oxigênio (O2)"
    )
    assert len(instances[0].input.text) == 1163

    assert instances[0].references == [
        Reference(
            output=Output(
                text="reduzir o calor irradiado pela Terra mediante a substituição da produção primária pela industrialização refrigerada. "  # noqa: E501
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="promover a queima da biomassa vegetal, responsável pelo aumento do efeito estufa devido à produção de CH4. "  # noqa: E501
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="reduzir o desmatamento, mantendo-se, assim, o potencial da vegetação em absorver o CO2 da atmosfera. "  # noqa: E501
            ),
            tags=[CORRECT_TAG],
        ),
        Reference(
            output=Output(
                text="aumentar a concentração atmosférica de H2O, molécula capaz de absorver grande quantidade de calor. "  # noqa: E501
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="remover moléculas orgânicas polares da atmosfera, diminuindo a capacidade delas de reter calor. "  # noqa: E501
            ),
            tags=[],
        ),
    ]
    assert instances[0].references[2].is_correct
