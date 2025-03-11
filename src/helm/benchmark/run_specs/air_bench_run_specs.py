from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("air_bench_2024")
def get_air_bench_2024_spec() -> RunSpec:
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        global_prefix="",
        global_suffix="",
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.0,
        stop_sequences=[],
    )
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.air_bench_scenario.AIRBench2024Scenario")
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.air_bench_annotator.AIRBench2024Annotator")]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.air_bench_metrics.AIRBench2024ScoreMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.air_bench_metrics.AIRBench2024BasicGenerationMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric"),
    ]
    return RunSpec(
        name="air_bench_2024",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["air_bench_2024"],
    )
