from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("oab_exams")
def get_enem_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.oab_exams_scenario.OABExamsScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Dê uma resposta selecionando uma letra entre as opções fornecidas. "
        "Se as opções forem A, B, C e D,"
        "sua resposta deve consistir em uma única letra que corresponde a resposta correta.\n"
        "Exemplo: Ao conselho da subseção compete\nA. representar a OAB no Conselho de Segurança do MERCOSUL."
        "\nB. fiscalizar as funções e atribuições do conselho seccional.\nC. instaurar e instruir processos "
        "disciplinares, para julgamento pelo Conselho Federal.\nD. receber pedido de inscrição nos quadros de "
        "advogado e estagiário, instruindo e emitindo parecer prévio, para decisão do conselho seccional.\n"
        "Resposta: D",
        input_noun="Pergunta",
        output_noun="Resposta",
    )

    return RunSpec(
        name="oab_exams",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["oab_exams"],
    )
