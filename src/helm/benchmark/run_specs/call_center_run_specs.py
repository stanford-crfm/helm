"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("call_center_summarization")
def get_call_center_summarization_spec(subset: str = "summarization") -> RunSpec:
    from helm.benchmark.annotation.call_center_annotator import CallCenterSummarizationAnnotator

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.call_center_scenario.CallCenterSummarizationScenario",
        args={"subset": subset},
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

    group = "call_center_summarization" if subset == "summarization" else f"call_center_summarization_{subset}"

    return RunSpec(
        name="call_center_summarization",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=[group],
    )


@run_spec_function("call_center_summarization_pairwise_comparison")
def get_call_center_summarization_pairwise_comparison_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.call_center_scenario.CallCenterSummarizationPairwiseComparisonScenario",
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
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.call_center_annotator.CallCenterSummarizationPairwiseComparisonAnnotator"  # noqa: E501
        )
    ]

    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(
            class_name="helm.benchmark.metrics.annotation_metrics.AnnotationNumericMetric",
            args={"annotator_name": "call_center_summarization_pairwise_comparison", "key": "score"},
        )
    ]

    return RunSpec(
        name="call_center_summarization_pairwise_comparison",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["call_center_summarization_pairwise_comparison"],
    )


@run_spec_function("call_center_summarization_key_points_recall")
def get_call_center_summarization_key_points_recall_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.call_center_scenario.CallCenterSummarizationKeyPointsRecallScenario",
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
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.call_center_annotator.CallCenterSummarizationKeyPointsRecallAnnotator"
        )
    ]

    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(
            class_name="helm.benchmark.metrics.annotation_metrics.AnnotationNumericMetric",
            args={"annotator_name": "call_center_summarization_key_points_recall", "key": "score"},
        )
    ]

    return RunSpec(
        name="call_center_summarization_key_points_recall",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["call_center_summarization_key_points_recall"],
    )


@run_spec_function("helpdesk_call_summarization")
def get_helpdesk_call_summarization_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.helpdesk_call_summarization_scenario.HelpdeskCallSummarizationScenario",
    )
    annotator_specs = annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.helpdesk_call_summarization_annotator.HelpdeskCallSummarizationAnnotator"  # noqa: E501
        )
    ]

    instructions = "The following is a call transcript of a call between a compnay's employee and the company's IT helpdesk. Summarize the call transcript in under 200 words."  # noqa: E501

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

    # annotator_specs = annotator_specs = [
    #     AnnotatorSpec(class_name="helm.benchmark.annotation.call_center_annotator.CallCenterSummarizationAnnotator")
    # ]
    annotation_metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.helpdesk_call_summarization_metrics.HelpdeskCallSummarizationMetric",
        ),
    ]

    metric_specs = get_basic_metric_specs([]) + annotation_metric_specs

    group = "helpdesk_call_summarization"

    return RunSpec(
        name="helpdesk_call_summarization",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=[group],
    )
