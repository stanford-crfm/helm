"""Run spec functions for the HELM Finance leaderboard.

Website: https://crfm.stanford.edu/helm/finance/"""

from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("fin_qa")
def get_fin_qa_spec() -> RunSpec:
    from helm.benchmark.scenarios.fin_qa_scenario import INSTRUCTIONS

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.fin_qa_scenario.FinQAScenario", args={})
    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS, input_noun=None, output_noun="Program", max_tokens=100
    )
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.fin_qa_metrics.FinQAMetric")
    ]
    return RunSpec(
        name="fin_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["fin_qa"],
    )


@run_spec_function("financebench")
def get_financebench_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.financebench_scenario.FinanceBenchScenario", args={}
    )
    adapter_spec = get_generation_adapter_spec(max_tokens=100)
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.financebench_annotator.FinanceBenchAnnotator")
    ]
    metric_specs = get_basic_metric_specs([])
    return RunSpec(
        name="financebench",
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["financebench"],
    )
