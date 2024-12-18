from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs, get_classification_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("tweetsentbr")
def get_tweetsentbr_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.tweetsentbr_scenario.TweetSentBRScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="""Classifique o tweet como "Positivo", "Neutro" ou "Negativo".

        Tweet: vocês viram a novela hoje?
        Classe: Neutro

        Tweet: que vontade de comer pizza
        Classe: Neutro
        """,
        input_noun="Tweet",
        output_noun="Classe",
    )

    return RunSpec(
        name="tweetsentbr",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["tweetsentbr"],
    )
