from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("enem_challenge")
def get_enem_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.enem_challenge_scenario.ENEMChallengeScenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Dê uma resposta selecionando uma letra entre as opções fornecidas. "
        "Se as opções forem A, B, C, D e E, "
        "sua resposta deve consistir em uma única letra que corresponde a resposta correta.\n"
        "Exemplo: Qual é a capital da França?\nA. Londres\nB. Paris\nC. Roma\nD. Berlim\nE. Sydney\n"
        "Resposta: B",
        input_noun="Pergunta",
        output_noun="Resposta",
    )

    return RunSpec(
        name="enem_challenge",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["enem_challenge"],
    )
