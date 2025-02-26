from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("bird_sql")
def get_bird_sql_dev_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.bird_sql_scenario.BIRDSQLScenario")

    adapter_spec = get_generation_adapter_spec(
        input_noun=None,
        output_noun=None,
        max_tokens=1024,
        stop_sequences=[],
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.bird_sql_annotator.BirdSQLAnnotator")]

    return RunSpec(
        name="bird_sql",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=get_exact_match_metric_specs()
        + [MetricSpec(class_name="helm.benchmark.metrics.bird_sql_metrics.BirdSQLMetric")],
        groups=["bird_sql"],
    )


@run_spec_function("spider")
def get_spider_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.spider_scenario.SpiderScenario")

    adapter_spec = get_generation_adapter_spec(
        input_noun=None,
        output_noun=None,
        max_tokens=1024,
        stop_sequences=[],
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.spider_annotator.SpiderAnnotator")]

    return RunSpec(
        name="spider",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=get_exact_match_metric_specs()
        + [MetricSpec(class_name="helm.benchmark.metrics.spider_metrics.SpiderMetric")],
        groups=["spider"],
    )
