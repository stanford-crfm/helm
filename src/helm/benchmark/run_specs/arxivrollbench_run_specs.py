from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_joint_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arxivrollbench")
def get_arxivrollbench_spec(
    release: str = "all",
    domain: str = "all",
    task_type: str = "all",
    split: str = "compact",
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arxivrollbench_scenario.ArxivRollBenchScenario",
        args={
            "release": release,
            "domain": domain,
            "task_type": task_type,
            "split": split,
        },
    )
    adapter_spec = get_multiple_choice_joint_adapter_spec(
        instructions="Answer the following scientific text reasoning questions with a single letter only.",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=0,
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs()
    return RunSpec(
        name=f"arxivrollbench:release={release},domain={domain},task_type={task_type},split={split}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["arxivrollbench"],
    )
