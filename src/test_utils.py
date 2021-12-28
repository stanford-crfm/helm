from scenario import ScenarioSpec
from adapter import AdapterSpec
from metric import MetricSpec
from runner import RunSpec


def get_scenario_spec1():
    return ScenarioSpec(
        class_name="simple_scenarios.SimpleScenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10,},
    )


def get_adapter_spec1():
    return AdapterSpec(
        instructions="Please solve the following problem.",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )


def get_run_spec1():
    """An run spec for debugging."""
    return RunSpec(
        scenario=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metrics=[MetricSpec(class_name="basic_metrics.BasicMetric", args={}),],
    )
