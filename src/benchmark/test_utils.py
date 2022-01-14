from typing import List

from .scenario import ScenarioSpec
from .adapter import AdapterSpec, ADAPT_MULTIPLE_CHOICE, ADAPT_GENERATION
from .metric import MetricSpec
from .runner import RunSpec


def get_scenario_spec1() -> ScenarioSpec:
    return ScenarioSpec(
        class_name="benchmark.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )


def get_scenario_spec_tiny():
    return ScenarioSpec(
        class_name="benchmark.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 2, "num_test_instances": 2},
    )


def get_adapter_spec1() -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )


def get_basic_metrics() -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={})]


############################################################


def get_run_spec1() -> RunSpec:
    """An run spec for debugging."""
    return RunSpec(
        name="run1", scenario=get_scenario_spec1(), adapter_spec=get_adapter_spec1(), metrics=get_basic_metrics()
    )


def get_mmlu_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.mmlu_scenario.MMLUScenario", args={"subject": subject},)

    def format(subject: str):
        return subject.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        instructions=f"The following are multiple choice questions (with answers) about {format(subject)}.",
        input_prefix="",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=1000,
        num_outputs=10,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0,
    )

    return RunSpec(
        name=f"mmlu_subject={subject}", scenario=scenario, adapter_spec=adapter_spec, metrics=get_basic_metrics()
    )
