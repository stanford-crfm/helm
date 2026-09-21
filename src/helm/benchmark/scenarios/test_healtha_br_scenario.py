import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.healthqa_br_scenario import HEALTHQA_BR_Scenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT, CORRECT_TAG, Output, Reference


@pytest.mark.scenarios
def test_healthqa_br_instance():
    scenario = HEALTHQA_BR_Scenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    instance = instances[35]

    assert instance.split == TEST_SPLIT

    assert instance.input.text.startswith("Homem de 22 anos de idade procura a Unidade Básica")

    assert instance.references == [
        Reference(
            output=Output(
                text="administração de relaxante muscular, colocando o paciente em posição de Trendelenburg, com "
                "tentativa de redução do volume."
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="encaminhamento do paciente ao Serviço de Urgência do Hospital com o pedido de avaliação "
                "imediata do cirurgião."
            ),
            tags=[CORRECT_TAG],
        ),
        Reference(
            output=Output(
                text="tentativa de redução manual do aumento de volume da região inguinescrotal para a cavidade "
                "abdominal."
            ),
            tags=[],
        ),
        Reference(
            output=Output(
                text="transiluminação do escroto para tentar diferenciar hérnia inguinal de hidrocele comunicante."
            ),
            tags=[],
        ),
        Reference(
            output=Output(text="prescrição de antiemético e solicitação de ecografia da região inguinescrotal."),
            tags=[],
        ),
    ]

    correct_refs = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
    assert len(correct_refs) == 1

    assert instance.references[1].is_correct
