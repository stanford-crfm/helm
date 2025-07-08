from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("healthqa_br")
def get_healthqa_br_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.healthqa_br_scenario.HEALTHQA_BR_Scenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="""
        Escolha a alternativa correta para as questões de medicina (responda apenas com a letra).
        Exemplo de Pergunta com a resposta:
        Qual dos seguintes órgãos é responsável pela produção da insulina no corpo humano?
        A) Fígado
        B) Rins
        C) Pâncreas
        D) Baço
        E) Coração

        Resposta correta: C

        A partir disso, responda:
        """,
        input_noun="Pergunta",
        output_noun="Resposta",
    )

    return RunSpec(
        name="healthqa_br",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["healthqa_br"],
    )
