from typing import Any, Dict, List

from helm.benchmark.adaptation.common_adapter_specs import get_completion_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_generative_harms_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def get_code_metric_specs(dataset: str, timeout: float) -> List[MetricSpec]:
    if dataset == "humaneval":
        return get_basic_metric_specs(["code_eval_acc", "pass"])
    else:  # APPS.
        args: Dict[str, Any] = {"names": ["test_avg", "strict_acc"], "timeout": timeout}
        return [MetricSpec(class_name="helm.benchmark.metrics.code_metrics.APPSMetric", args=args)]


@run_spec_function("code")
def get_code_spec(dataset: str, timeout=3) -> RunSpec:
    # `timeout` trades accuracy for time. Used exclusively for APPS. Default from original APPS codebase.
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.code_scenario.CodeScenario", args={"dataset": dataset}
    )

    if dataset == "humaneval":
        adapter_spec = get_completion_adapter_spec(
            temperature=0.2,
            # Taken from the original OpenAI paper to prevent the further generation of irrelevant classes/functions
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint"],
            max_tokens=600,
        )
    else:  # apps.
        # Different in `stop_sequences`.
        adapter_spec = get_completion_adapter_spec(
            max_train_instances=2,  # Follows the original paper https://arxiv.org/pdf/2105.09938.pdf Appendix D.
            temperature=0.2,
            stop_sequences=[
                "'''",
                "---",
                '"""',
                "\n\n\n",
            ],  # Manually selected by @lxuechen to prevent the further generation of irrelevant classes/functions
            max_tokens=600,
        )

    return RunSpec(
        name=f"code:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_metric_specs(dataset, timeout) + get_generative_harms_metric_specs(),
        groups=[f"code_{dataset}"],
    )
