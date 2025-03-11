from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.reeval_parameters import ReevalParameters


from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
)


@run_spec_function("reeval_air_bench_2024")
def get_reeval_air_bench_2024_spec() -> RunSpec:
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
        reeval_parameters=ReevalParameters(model_ability=0.0, max_samples=50, metric_name="air_score"),
    )
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.air_bench_scenario.AIRBench2024Scenario")
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.air_bench_annotator.AIRBench2024Annotator")]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.air_bench_metrics.AIRBench2024ScoreMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.air_bench_metrics.AIRBench2024BasicGenerationMetric"),
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric"),
    ]
    return RunSpec(
        name="reeval_air_bench_2024",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["air_bench_2024"],
    )


@run_spec_function("reeval_mmlu")
def get_reeval_mmlu_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_scenario.MMLUScenario", args={"subject": subject}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.",
        input_noun="Question",
        output_noun="Answer",
        reeval_parameters=ReevalParameters(model_ability=0.0, max_samples=50, metric_name="exact_match"),
    )

    return RunSpec(
        name=f"reeval_mmlu:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu"],
    )
