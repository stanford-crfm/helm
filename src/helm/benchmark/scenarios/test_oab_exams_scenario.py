import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.oab_exams_scenario import OABExamsScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_oab_exams_scenario():
    oab_scenario = OABExamsScenario()
    with TemporaryDirectory() as tmpdir:
        instances = oab_scenario.get_instances(tmpdir)
    assert len(instances) == 2210
    assert instances[100].split == TEST_SPLIT

    assert instances[100].input.text.startswith(
        "O Congresso Nacional e suas respectivas Casas se reúnem anualmente para a atividade legislativa."
    )
    assert len(instances[100].input.text) == 178

    assert instances[100].references == [
        Reference(
            output=Output(
                text="Legislatura: o período compreendido entre 2 de fevereiro a 17 de julho e 1º de agosto a 22 de dezembro."  # noqa: E501
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="Sessão legislativa: os quatro anos equivalentes ao mandato dos parlamentares."  # noqa: E501
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="Sessão conjunta: a reunião da Câmara dos Deputados e do Senado Federal destinada, por exemplo, "
                "a conhecer do veto presidencial e sobre ele deliberar."  # noqa: E501
            ),
            tags=[CORRECT_TAG],
        ),
        Reference(
            output=Output(
                text="Sessão extraordinária: a que ocorre por convocação ou do Presidente do Senado Federal ou do "
                "Presidente da Câmara dos Deputados ou do Presidente da República e mesmo por requerimento da maioria "
                "dos membros de ambas as Casas para, excepcionalmente, inaugurar a "
                "sessão legislativa e eleger as respectivas mesas diretoras."  # noqa: E501
            ),
            tags=[],
        ),
    ]
    assert instances[100].references[2].is_correct
