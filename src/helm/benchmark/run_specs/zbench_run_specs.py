"""Run specs for zBench IVF (zReports HealthCare Data Lab), MedHELM category Administration and Workflow.

helm-run --run-entries zbench_ivf:condition=closed,model=openai/gpt-5-2025-08-07 --suite zbench --max-eval-instances 500
helm-run --run-entries zbench_ivf:condition=record,model=openai/gpt-5-2025-08-07 --suite zbench --max-eval-instances 500
helm-run --run-entries zbench_ivf:condition=record,layers=1+2+6,model=openai/gpt-5-2025-08-07 --suite zbench

`layers` keeps the listed layers only; the numbers are separated by `+` because run entry arguments split on commas.
"""

from typing import Optional, Union

from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

INSTRUCTIONS = (
    "Answer the question about the healthcare facility record. Give the figure exactly as the source publishes it, "
    "with any qualifier the source attaches (small sample, suppressed, no cycles); if the source publishes no figure "
    "for the cell, say exactly what it does publish."
)


@run_spec_function("zbench_ivf")
def get_zbench_ivf_spec(condition: str = "closed", layers: Optional[Union[str, int]] = None) -> RunSpec:
    layers_arg = str(layers) if layers not in (None, "") else None
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.zbench_scenario.ZBenchIVFScenario",
        args={"condition": condition, **({"layers": layers_arg} if layers_arg else {})},
    )
    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS,
        input_noun=None,
        newline_after_input_noun=False,
        output_noun="Answer",
        max_tokens=600,
        max_train_instances=0,
        stop_sequences=[],
    )
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.zbench_metrics.ZBenchMetric", args={})
    ] + get_basic_metric_specs([])
    return RunSpec(
        name="zbench_ivf:condition=%s%s" % (condition, (",layers=" + layers_arg) if layers_arg else ""),
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["zbench_ivf"],
    )
