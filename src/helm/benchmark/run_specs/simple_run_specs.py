"""Run spec functions for tutorials and for debugging."""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_joint_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("simple_mcqa")
def get_simple_mcqa_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.simple_scenarios.SimpleMCQAScenario")
    adapter_spec = get_multiple_choice_joint_adapter_spec(
        instructions="Answer the following questions with a single letter only.",
        input_noun="Question",
        output_noun="Answer",
    )
    metric_specs = get_exact_match_metric_specs()
    return RunSpec(
        name="simple_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["simple_mcqa"],
    )


@run_spec_function("simple_short_answer_qa")
def get_simple_short_answer_qa_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.simple_scenarios.SimpleShortAnswerQAScenario")
    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the following questions with a single word only.",
        input_noun="Question",
        output_noun="Answer",
    )
    # NOTE: Open ended generation metrics measure the amount of overlap
    # (e.g. ROUGE, BLEU, F1 word overlap) between the generated output
    # and the correct reference outputs.
    metric_specs = get_open_ended_generation_metric_specs()
    return RunSpec(
        name="simple_short_answer_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["simple_short_answer_qa"],
    )


@run_spec_function("simple_classification")
def get_simple_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.simple_scenarios.SimpleClassificationScenario")
    adapter_spec = get_generation_adapter_spec(
        instructions='Classify the following numbers by their parity. The classes are "Even" and "Odd".',
        input_noun="Number",
        output_noun="Parity",
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="simple_classification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["simple_classification"],
    )


@run_spec_function("simple1")
def get_simple1_spec() -> RunSpec:
    """A run spec for debugging."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.\n",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        model_deployment="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )
    return RunSpec(
        name="simple1",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_generation_metric_specs([]) + get_generic_metric_specs(),
        groups=[],
    )
