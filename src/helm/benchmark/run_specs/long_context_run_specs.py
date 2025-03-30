from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_open_ended_generation_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _get_long_context_generation_adapter_spec(max_tokens: int) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION,
        global_prefix="",
        global_suffix="",
        instructions="",
        input_prefix="",
        input_suffix="",
        reference_prefix="A. ",
        reference_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=0,
        num_outputs=1,
        temperature=0.0,
        max_tokens=max_tokens,
        stop_sequences=[],
    )


@run_spec_function("ruler_hotpotqa")
def get_ruler_hotpotqa_spec(max_num_words: int = 65536) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ruler_qa_scenarios.RULERHotpotQAScenario",
        args={
            "max_num_words": max_num_words,
        },
    )

    adapter_spec = _get_long_context_generation_adapter_spec(max_tokens=100)

    return RunSpec(
        name=f"ruler_hotpotqa:max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["ruler_hotpotqa"],
    )


@run_spec_function("ruler_squad")
def get_ruler_squad_spec(max_num_words: int = 65536) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ruler_qa_scenarios.RULERSQuADScenario",
        args={
            "max_num_words": max_num_words,
        },
    )

    adapter_spec = _get_long_context_generation_adapter_spec(max_tokens=100)

    return RunSpec(
        name=f"ruler_squad:max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["ruler_squad"],
    )


@run_spec_function("infinite_bench_sum")
def get_infinite_bench_sum_spec(min_num_words: int = 0, max_num_words: int = 65536) -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.infinite_bench_sum_scenario.InfiniteBenchSumScenario",
        args={
            "min_num_words": min_num_words,
            "max_num_words": max_num_words,
        },
    )

    # No official number for max tokens, the average output token is 1.1k according to the paper
    adapter_spec = _get_long_context_generation_adapter_spec(max_tokens=2000)
    metric_specs = get_basic_metric_specs(["rouge_l"])

    return RunSpec(
        name=f"infinite_bench_sum:min_num_words={min_num_words},max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["infinite_bench_sum"],
    )
