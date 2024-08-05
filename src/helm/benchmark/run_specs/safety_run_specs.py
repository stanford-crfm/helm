from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

from helm.benchmark.metrics.metric import MetricSpec


@run_spec_function("xstest")
def get_simple_safety_tests_spec() -> RunSpec:
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
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.xstest_scenario.XSTestScenario")
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.xstest_annotator.XSTestAnnotator")]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.safety_metrics.SafetyScoreMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.safety_metrics.SafetyBasicGenerationMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric"),
    ]
    return RunSpec(
        name="xstest",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["xstest"],
    )
