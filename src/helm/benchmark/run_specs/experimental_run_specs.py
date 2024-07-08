"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec, get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
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


@run_spec_function("verifiability_judgment_with_explanation")
def get_verifiability_judgment_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.verifiability_judgment_scenario.VerifiabilityJudgementScenario", args={}
    )

    instructions = (
        "You will be provided with a document delimited by triple quotes and a statement. "
        'Your task is to judge whether the provided document "fully supports", "partially supports" or "does not support" the statement. '  # noqa: E501
        'If the answer is "fully supports" or "partially supports", it must be annotated with direct citation from the document and/or explanation. '  # noqa: E501
        'Use the following format for output: {"answer": "..." , "citation": "..." , "explanation": "..."}'
    )
    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
        input_noun="Statement",
        # Add another new line before the output noun, since the source might have
        # newlines embedded in it.
        output_noun="\nJudgment",
        max_tokens=300,
        max_train_instances=0,
    )

    return RunSpec(
        name="verifiability_judgment_with_explanation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs([]),
        groups=["verifiability_judgment_with_explanation"],
    )
