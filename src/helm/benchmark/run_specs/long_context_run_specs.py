from helm.benchmark.adaptation.adapter_spec import ADAPT_CHAT, ADAPT_GENERATION, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_open_ended_generation_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
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
def get_ruler_hotpotqa_spec(max_num_words: int = 131072) -> RunSpec:
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
def get_ruler_squad_spec(max_num_words: int = 131072) -> RunSpec:
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


@run_spec_function("infinite_bench_en_qa")
def get_infinite_bench_en_qa_spec(max_num_words: int = 131072) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.infinite_bench_en_qa_scenario.InfiniteBenchEnQAScenario",
        args={
            "max_num_words": max_num_words,
        },
    )

    adapter_spec = _get_long_context_generation_adapter_spec(max_tokens=40)
    metric_specs = get_basic_metric_specs(["rouge_l"])

    return RunSpec(
        name="infinite_bench_en_qa:max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["infinite_bench_en_qa"],
    )


@run_spec_function("infinite_bench_en_sum")
def get_infinite_bench_en_sum_spec(max_num_words: int = 131072) -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.infinite_bench_en_sum_scenario.InfiniteBenchEnSumScenario",
        args={
            "max_num_words": max_num_words,
        },
    )

    adapter_spec = _get_long_context_generation_adapter_spec(max_tokens=1200)
    metric_specs = get_basic_metric_specs(["rouge_l"])

    return RunSpec(
        name=f"infinite_bench_en_sum:max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["infinite_bench_en_sum"],
    )


@run_spec_function("openai_mrcr")
def get_openai_mrcr_spec(needles: int, max_num_words: int = 131072) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.openai_mrcr_scenario.OpenAIMRCRScenario",
        args={"needles": needles, "max_num_words": max_num_words},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_CHAT, input_prefix="", output_prefix="", max_tokens=2000, num_outputs=1, temperature=0.0
    )
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.openai_mrcr_metrics.OpenAIMRCRMetric")
    ]

    return RunSpec(
        name=f"openai_mrcr:needles={needles},max_num_words={max_num_words}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["openai_mrcr"],
    )
