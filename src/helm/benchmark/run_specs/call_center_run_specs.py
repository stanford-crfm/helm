"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("call_center_summarization")
def get_xsum_summarization_spec(revision: str = "main") -> RunSpec:
    from helm.benchmark.annotation.call_center_annotator import CallCenterSummarizationAnnotator

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.call_center_scenario.CallCenterSummarizationScenario",
        args={"revision": revision},
    )

    instructions = "Summarize the call transcript in under 10 sentences."

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=instructions,
        input_prefix="### Call Transcript\n",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        temperature=0.0,
        max_tokens=512,
        num_outputs=1,
    )

    annotator_specs = annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.call_center_annotator.CallCenterSummarizationAnnotator")
    ]
    annotation_metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.annotation_metrics.AnnotationLikertScaleMetric",
            args={
                "annotator_name": CallCenterSummarizationAnnotator.name,
                "key": criterion,
                "min_score": 1,
                "max_score": 5,
            },
        )
        for criterion in CallCenterSummarizationAnnotator.CRITERIA
    ]

    metric_specs = get_basic_metric_specs([]) + annotation_metric_specs

    return RunSpec(
        name="call_center_summarization",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["call_center_summarization"],
    )
