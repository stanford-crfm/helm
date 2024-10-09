"""Run spec functions for HELM Instruct.

Website: https://crfm.stanford.edu/helm/instruct/"""

from typing import List

from helm.benchmark.adaptation.common_adapter_specs import get_instruct_adapter_spec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def get_instruction_following_critique_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.instruction_following_critique_metrics"
            ".InstructionFollowingCritiqueMetric",
            # noqa E501
            args={"num_respondents": num_respondents},
        )
    ]


@run_spec_function("self_instruct")
def get_self_instruct_spec(num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.self_instruct_scenario.SelfInstructScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name="self_instruct",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["self_instruct"],
    )


@run_spec_function("vicuna")
def get_vicuna_spec(num_respondents: int, category: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vicuna_scenario.VicunaScenario",
        args={"category": category},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"vicuna:category={category}",  # TODO: add args
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["vicuna"],
    )


@run_spec_function("grammar")
def get_grammar_spec(num_respondents: int, path: str, tags: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.grammar_scenario.GrammarScenario",
        args={"path": path, "tags": tags},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"grammar:path={path},tags={tags}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["grammar"],
    )


@run_spec_function("open_assistant")
def get_open_assistant_spec(num_respondents: int, language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.open_assistant_scenario.OpenAssistantScenario",
        args={"language": language},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"open_assistant:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["open_assistant"],
    )


@run_spec_function("koala")
def get_koala_spec(num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.koala_scenario.KoalaScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name="koala",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["koala"],
    )


@run_spec_function("anthropic_hh_rlhf")
def get_anthropic_hh_rlhf_spec(num_respondents: int, subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.anthropic_hh_rlhf_scenario.AnthropicHHRLHFScenario",
        args={"subset": subset},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"anthropic_hh_rlhf:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["anthropic_hh_rlhf"],
    )
