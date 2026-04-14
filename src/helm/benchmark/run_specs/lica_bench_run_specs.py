"""Run spec functions for `lica-bench` (graphic design VLM benchmarks)."""

from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _lica_bench_adapter_spec() -> AdapterSpec:
    """Generous token budget for multi-line or structured answers (e.g. SVG snippets, lists)."""
    return AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        global_prefix="",
        instructions="Follow the prompt and answer completely. Use the format implied by the prompt.",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        instance_prefix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=2048,
        stop_sequences=[],
        temperature=0.0,
        random=None,
    )


def _lica_bench_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        [
            "exact_match",
            "quasi_exact_match",
            "quasi_leave_articles_exact_match",
            "f1_score",
            "rouge_l",
            "bleu_1",
            "bleu_4",
            "cider",
        ]
    )


@run_spec_function("lica_bench")
def get_lica_bench_spec(
    benchmark_id: str,
    dataset_root: str = "",
    max_instances: Optional[int] = None,
) -> RunSpec:
    """
    Run a single lica-bench task (``category-1``, ``svg-2``, …).

    :param benchmark_id: Task id as defined in lica-bench (see ``python -m design_benchmarks`` or
        ``scripts/run_benchmarks.py --list`` in the lica-bench repo).
    :param dataset_root: Path to the unpacked ``lica-benchmarks-dataset`` directory. If empty,
        uses the ``LICA_BENCH_DATASET_ROOT`` environment variable.
    :param max_instances: Optional cap on the number of instances (for dry runs).
    """
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.lica_bench_scenario.LicaBenchScenario",
        args={
            "benchmark_id": benchmark_id,
            "dataset_root": dataset_root,
            "max_instances": max_instances,
        },
    )

    adapter_spec = _lica_bench_adapter_spec()
    metric_specs = _lica_bench_metric_specs()

    dr_part = f",dataset_root={dataset_root}" if dataset_root else ""
    mi_part = f",max_instances={max_instances}" if max_instances is not None else ""
    return RunSpec(
        name=f"lica_bench:benchmark_id={benchmark_id}{dr_part}{mi_part}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lica_bench", f"lica_bench_{benchmark_id}"],
    )
