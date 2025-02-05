from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs, get_classification_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("imdb_ptbr")
def get_tweetsentbr_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.imdb_ptbr_scenario.IMDB_PTBRScenario", args={})

    adapter_spec = get_generation_adapter_spec(
        instructions="""Classifique a resenha do usuário sobre o filme como "positivo" ou "negativo".

        Resenha: Tudo sobre o filme é maravilhoso. Atuações, trilha sonora, fotografia. Amei tudo!
        Classe: positivo

        Resenha: Achei um filme bem fraco, não gostei da história.
        Classe: negativo
        """,
        input_noun="Resenha",
        output_noun="Classe",
    )

    return RunSpec(
        name="imdb_ptbr",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["imdb_ptbr"],
    )
