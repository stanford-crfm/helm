"""Run spec functions for the HELM Finance leaderboard.

Website: https://crfm.stanford.edu/helm/finance/"""

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


@run_spec_function("fin_qa")
def get_fin_qa_spec() -> RunSpec:
    from helm.benchmark.scenarios.fin_qa_scenario import INSTRUCTIONS

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.fin_qa_scenario.FinQAScenario", args={})
    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS, input_noun=None, output_noun="Program", max_tokens=100
    )
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.fin_qa_metrics.FinQAMetric")
    ]
    return RunSpec(
        name="fin_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["fin_qa"],
    )


@run_spec_function("financebench")
def get_financebench_spec() -> RunSpec:
    instructions = (
        "Answer only the last question using the given evidence. "
        "Respond with only a single paragraph, sentence or sentence fragment.\n"
    )
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.financebench_scenario.FinanceBenchScenario", args={}
    )
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=instructions,
        input_prefix="\n",
        input_suffix="\n",
        output_prefix="\nAnswer: ",
        output_suffix="\n",
        instance_prefix="\n###\n",
        num_outputs=1,
        max_tokens=300,
        temperature=0.0,
        stop_sequences=["###"],
    )
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.financebench_annotator.FinanceBenchAnnotator")
    ]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(
            class_name="helm.benchmark.metrics.annotation_metrics.AnnotationLabelMetric",
            args={
                "annotator_name": "financebench",
                "key": "label",
                "labels": ["correct_answer", "incorrect_answer", "failure_to_answer"],
            },
        )
    ]
    return RunSpec(
        name="financebench",
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["financebench"],
    )


@run_spec_function("banking77")
def get_banking77_spec() -> RunSpec:
    from helm.benchmark.scenarios.raft_scenario import get_raft_instructions
    from helm.benchmark.scenarios.banking77_scenario import Banking77Scenario

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.banking77_scenario.Banking77Scenario", args={})

    # Use same AdapterSpec and instruction prompts as the RAFT implementation of BANKING77,
    # with a slight modification to the instruction prompt for instruction-following models.
    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), Banking77Scenario.name)
    instructions = get_raft_instructions("banking_77", scenario_cache_path).replace(
        "\n", " Answer with only the label for the last query.\n", 1
    )
    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
        input_noun=None,
        output_noun="Label",
        max_tokens=30,  # at most ~50 characters per label
    )

    # Not using get_classification_metric_specs() / ClassificationMetric because BANKING77 has too many classes,
    # so F1 scores don't make sense. The original paper uses accuracy instead.
    metric_specs = get_exact_match_metric_specs()
    return RunSpec(
        name="banking77",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["banking77"],
    )
