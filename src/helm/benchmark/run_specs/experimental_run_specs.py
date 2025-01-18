"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec, get_generation_adapter_spec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ci_mcqa")
def get_ci_mcqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.ci_mcqa_scenario.CIMCQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "Give a letter answer among the options given. "
            "For example, if the options are A, B, C, D, E, and F, "
            "your answer should consist of the single letter that corresponds to the correct answer."
        ),
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="ci_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["CIMCQA"],
    )


@run_spec_function("ewok")
def get_ewok_spec(domain: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ewok_scenario.EWoKScenario", args={"domain": domain}
    )

    instructions = """# INSTRUCTIONS

In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense considering the scenario that follows. The contexts will be numbered "1" or "2". You must answer using "1" or "2" in your response.
"""  # noqa: E501
    input_prefix = """# TEST EXAMPLE

## Scenario
\""""
    input_suffix = """\"

## Contexts
"""
    output_prefix = """
## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
"""

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=instructions,
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        reference_prefix='1. "',
        reference_suffix='"\n',
        output_prefix=output_prefix,
        output_suffix="\n",
        max_train_instances=2,
        num_outputs=1,
        max_tokens=2,
        temperature=0.0,
        stop_sequences=["\n\n"],
        sample_train=True,
    )

    return RunSpec(
        name=f"ewok:domain={domain}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["ewok", f"ewok_{domain}"],
    )


@run_spec_function("autobencher_capabilities")
def get_autobencher_capabilities_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.autobencher_capabilities_scenario.AutoBencherCapabilitiesScenario",
        args={"subject": subject},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=("Output just with the final answer to the question."),
        input_noun="Question",
        output_noun="Answer",
        max_tokens=100,
    )
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.autobencher_capabilities_annotator.AutoBencherCapabilitiesAnnotator"
        )
    ]
    annotator_metric_spec = MetricSpec(
        class_name="helm.benchmark.metrics.annotation_metrics.AnnotationNumericMetric",
        args={
            "annotator_name": "autobencher_capabilities",
            "key": "score",
        },
    )

    return RunSpec(
        name="autobencher_capabilities",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=get_exact_match_metric_specs() + [annotator_metric_spec],
        groups=["autobencher_capabilities"],
    )


@run_spec_function("autobencher_safety")
def get_autobencher_safety_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.autobencher_safety_scenario.AutoBencherSafetyScenario",
    )

    adapter_spec = adapter_spec = AdapterSpec(
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
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.autobencher_safety_annotator.AutoBencherSafetyAnnotator")
    ]
    annotator_metric_spec = MetricSpec(
        class_name="helm.benchmark.metrics.annotation_metrics.AnnotationNumericMetric",
        args={
            "annotator_name": "autobencher_safety",
            "key": "score",
        },
    )

    return RunSpec(
        name="autobencher_safety",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=get_exact_match_metric_specs() + [annotator_metric_spec],
        groups=["autobencher_safety"],
    )


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
