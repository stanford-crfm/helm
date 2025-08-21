from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("bluex")
def get_bluex_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.bluex_scenario.BLUEXScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="""
        Escolha a alternativa correta para as questões de vestibulares (responda apenas com a letra).
        Exemplo de Pergunta com a resposta:
        Em um romance narrado em primeira pessoa, o narrador participa dos acontecimentos da trama,
        relatando suas próprias experiências e sentimentos. Qual alternativa apresenta essa característica?

        (A) Narrador onisciente que conhece os pensamentos de todas as personagens.
        (B) Narrador que descreve os fatos de forma imparcial, sem envolvimento emocional.
        (C) Narrador-personagem que vivencia e relata os eventos da história.
        (D) Narrador observador que apenas registra as ações visíveis.
        (E) Narrador em segunda pessoa que se dirige constantemente ao leitor.

        Resposta correta: C

        A partir disso, responda:
        """,
        input_noun="Pergunta",
        output_noun="Resposta",
    )

    return RunSpec(
        name="bluex",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bluex"],
    )
