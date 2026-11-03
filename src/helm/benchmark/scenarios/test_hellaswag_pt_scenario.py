from tempfile import TemporaryDirectory

import pytest

from helm.benchmark.scenarios.hella_swag_pt_scenario import HellaSwag_PT_Scenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


@pytest.mark.scenarios
def test_hellaswag_pt_scenario():
    scenario = HellaSwag_PT_Scenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances, "The scenario must return evaluation instances"

    for instance in instances:
        assert instance.split == TEST_SPLIT
        assert instance.input.text.strip(), "The context must not be empty"
        assert all(reference.output.text.strip() for reference in instance.references)
        assert sum(reference.is_correct for reference in instance.references) == 1

    # Locate a known example by context rather than depending on dataset order.
    # Source ID: hellaswag/validation/1559. Its label is "3" (fourth ending).
    expected_context = "Uma mulher de terno vermelho está falando atrás de uma mesa. um homem"
    matching_instances = [instance for instance in instances if instance.input.text == expected_context]
    assert len(matching_instances) == 1, "Expected exactly one matching example"
    instance = matching_instances[0]

    assert instance.references == [
        Reference(
            output=Output(text="pega bolas de um suporte na parede."),
            tags=[],
        ),
        Reference(
            output=Output(text="está aspirando uma cadeira plástica preta."),
            tags=[],
        ),
        Reference(
            output=Output(text="espirra água em uma pia branca pequena."),
            tags=[],
        ),
        Reference(
            output=Output(text="está em pé segurando o pé atrás das costas."),
            tags=[CORRECT_TAG],
        ),
    ]
    assert instance.references[3].is_correct
