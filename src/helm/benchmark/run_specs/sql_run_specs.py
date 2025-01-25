from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("czech_bank_qa")
def get_czech_bank_qa_spec(config_name: str = "berka_queries_1024_2024_12_18") -> RunSpec:
    from helm.benchmark.scenarios.czech_bank_qa_scenario import CzechBankQAScenario

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.czech_bank_qa_scenario.CzechBankQAScenario",
        args={"config_name": config_name},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=CzechBankQAScenario.INSTRUCTIONS,
        input_noun="Instruction",
        output_noun="SQL Query",
        max_tokens=512,
        stop_sequences=["\n\n"],
    )

    return RunSpec(
        name="czech_bank_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs([])
        + [MetricSpec(class_name="helm.benchmark.metrics.czech_bank_qa_metrics.CzechBankQAMetrics", args={})],
        annotators=[AnnotatorSpec("helm.benchmark.annotation.czech_bank_qa_annotator.CzechBankQAAnnotator")],
        groups=["czech_bank_qa"],
    )


@run_spec_function("bird_sql_dev")
def get_bird_sql_dev() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.bird_sql_dev_scenario.BIRDSQLDevScenario")

    adapter_spec = get_generation_adapter_spec(
        input_noun=None,
        output_noun=None,
        max_tokens=1024,
        stop_sequences=[],
    )

    return RunSpec(
        name="bird_sql_dev",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bird_sql_dev"],
    )
