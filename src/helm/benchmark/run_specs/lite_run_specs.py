"""Run spec functions for the HELM Lite leaderboard.

Website: https://crfm.stanford.edu/helm/lite/"""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_machine_translation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


@run_spec_function("narrative_qa")
def get_narrativeqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.narrativeqa_scenario.NarrativeQAScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=100,  # max 30 words
    )

    return RunSpec(
        name="narrative_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["narrative_qa"],
    )


@run_spec_function("natural_qa")
def get_natural_qa_spec(mode: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={"mode": mode}
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Question" if mode == "closedbook" else None,
        output_noun="Answer",
        max_tokens=300,  # answers are at most 65 words
    )

    return RunSpec(
        name=f"natural_qa:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=[f"natural_qa_{mode}"],
    )


@run_spec_function("commonsense")
def get_commonsense_spec(dataset: str, method: str) -> RunSpec:
    from helm.benchmark.scenarios.commonsense_scenario import (
        CommonSenseQAScenario,
        HellaSwagScenario,
        OpenBookQA,
        PiqaScenario,
        SiqaScenario,
    )

    # TODO Split these into their own run_spec_function.
    if dataset == HellaSwagScenario.name:
        scenario_spec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.commonsense_scenario.HellaSwagScenario", args={}
        )
    elif dataset == OpenBookQA.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.OpenBookQA", args={})
    elif dataset == CommonSenseQAScenario.name:
        scenario_spec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.commonsense_scenario.CommonSenseQAScenario", args={}
        )
    elif dataset == SiqaScenario.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.SiqaScenario", args={})
    elif dataset == PiqaScenario.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.PiqaScenario", args={})
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers) about common sense.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"commonsense:dataset={dataset},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[dataset],
    )


@run_spec_function("mmlu")
def get_mmlu_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_scenario.MMLUScenario", args={"subject": subject}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"mmlu:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu"],
    )


@run_spec_function("gsm")
def get_gsm_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})

    # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
    adapter_spec = get_generation_adapter_spec(
        input_noun="Q",
        output_noun="A",
        max_train_instances=5,  # Due to limited context and long example length
        max_tokens=400,  # The paper uses 400 tokens as the max sample length
        stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
    )

    return RunSpec(
        name="gsm",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_generation_metric_specs(["exact_match_indicator", "final_number_exact_match"])
        + get_generic_metric_specs()
        + get_generative_harms_metric_specs(),
        groups=["gsm"],
    )


@run_spec_function("math")
def get_math_spec(
    subject: str,
    level: str,
    use_official_examples: str = "False",
    use_chain_of_thought: str = "False",
) -> RunSpec:
    # Convert to bools and remove the str versions
    use_official_examples_bool: bool = use_official_examples.lower() == "true"
    use_chain_of_thought_bool: bool = use_chain_of_thought.lower() == "true"
    del use_official_examples
    del use_chain_of_thought

    if use_chain_of_thought_bool:
        assert not use_official_examples_bool, "Cannot use official examples when use_chain_of_thought is True."
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.math_scenario.MATHScenario",
        args={
            "subject": subject,
            "level": level,
            "use_official_examples": use_official_examples_bool,
            "use_chain_of_thought": use_chain_of_thought_bool,
        },
    )

    if use_chain_of_thought_bool:  # Include the solution in the output as per https://arxiv.org/abs/2201.11903
        output_prefix = "Answer: "  # Don't include LaTeX '$' delimiters
        output_suffix = "\n"
        instance_prefix = "###\n"  # Don't include LaTeX '$' delimiters
        max_tokens = 400  # Increase the number of tokens to generate
        stop_sequences = ["###"]  # Break at the next instance; extraneous output will be stripped out
        groups = ["math_chain_of_thought"]
    else:
        output_prefix = "Answer: $"
        output_suffix = "$\n"
        instance_prefix = "###\n"
        max_tokens = 20
        stop_sequences = ["$"]  # Break at the nearest LaTeX closing delimiter
        groups = ["math_regular"]

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Given a mathematics problem, determine the answer. Simplify your answer as much as possible.\n",
        max_train_instances=8,
        num_outputs=1,
        temperature=0.0,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        input_prefix="Problem: ",
        input_suffix="\n",
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        instance_prefix=instance_prefix,
    )

    return RunSpec(
        name=f"math:subject={subject},level={level},"
        f"use_official_examples={use_official_examples_bool},use_chain_of_thought={use_chain_of_thought_bool}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(
            ["math_equiv_chain_of_thought" if use_chain_of_thought_bool else "math_equiv"]
        )
        + get_generative_harms_metric_specs(),
        groups=groups,
    )


@run_spec_function("legalbench")
def get_legalbench_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.legalbench_scenario import (
        LegalBenchScenario,
        get_legalbench_instructions,
        get_legalbench_output_nouns,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legalbench_scenario.LegalBenchScenario", args={"subset": subset}
    )
    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), LegalBenchScenario.name)
    adapter_spec = get_generation_adapter_spec(
        instructions=get_legalbench_instructions(subset, scenario_cache_path),
        input_noun=None,
        output_noun=get_legalbench_output_nouns(subset, scenario_cache_path),
        max_tokens=30,  # at most ~50 characters per label,
        max_train_instances=5,  # Use 5 for all subsets
    )

    return RunSpec(
        name=f"legalbench:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["legalbench"],
    )


@run_spec_function("med_qa")
def get_med_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.med_qa_scenario.MedQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="The following are multiple choice questions (with answers) about medicine.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="med_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["med_qa"],
    )


@run_spec_function("wmt_14")
def get_wmt_14_spec(language_pair: str, max_train_instances: int = 1) -> RunSpec:
    FULL_LANGUAGE_NAMES = {
        "cs": "Czech",
        "de": "German",
        "fr": "French",
        "hi": "Hindi",
        "ru": "Russian",
        "en": "English",
    }
    source_language, target_language = language_pair.split("-")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.wmt_14_scenario.WMT14Scenario",
        args={"source_language": source_language, "target_language": target_language},
    )

    adapter_spec = get_machine_translation_adapter_spec(
        source_language=FULL_LANGUAGE_NAMES[source_language],
        target_language=FULL_LANGUAGE_NAMES[target_language],
        max_train_instances=max_train_instances,
    )

    return RunSpec(
        name=f"wmt_14:language_pair={language_pair}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["wmt_14"],
    )
