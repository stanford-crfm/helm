from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec


@run_spec_function("ts_guessing_contamination")
def get_contamination_spec(dataset: str, strategy: str, language: str) -> RunSpec:
    valid_strategies = {"ts_guessing_question_base", "ts_guessing_question_multichoice"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(valid_strategies)}")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ts_guessing_contamination.ts_guessing_contamination_scenario.TSGuessingContaminationScenario",
        args={"dataset": dataset, "strategy": strategy, "language": language},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="",
        input_noun="Input",
        output_noun="Result",
        max_tokens=100,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    # 1. Apply the custom metric (with official cleaning) to ALL TS-Guessing strategies
    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.ts_guessing_contamination_metrics.TSGuessingMetric",
            args={"language": language},
        )
    ]

    # 2. Add the generic ones as a comparison baseline
    metric_specs.extend(
        get_basic_metric_specs(
            [
                "exact_match",
                "quasi_exact_match",
                "prefix_exact_match",
                "quasi_prefix_exact_match",
                "rouge_l",
            ]
        )
    )

    return RunSpec(
        name=f"ts_guessing_contamination:dataset={dataset},strategy={strategy},language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["ts_guessing_contamination"],
    )
