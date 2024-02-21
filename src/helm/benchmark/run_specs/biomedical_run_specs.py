from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_generative_harms_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("covid_dialog")
def get_covid_dialog_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.covid_dialog_scenario.COVIDDialogScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Generate a response given a patient's questions and concerns.",
        input_noun="Patient",
        output_noun="Doctor",
        max_tokens=128,
    )

    return RunSpec(
        name="covid_dialog",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["COVIDDialog"],
    )


@run_spec_function("me_q_sum")
def get_me_q_sum_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.me_q_sum_scenario.MeQSumScenario", args={})

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=128,
        temperature=0.3,
    )

    return RunSpec(
        name="me_q_sum",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MeQSum"],
    )


@run_spec_function("med_dialog")
def get_med_dialog_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_dialog_scenario.MedDialogScenario", args={"subset": subset}
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=128,
        temperature=0.3,
    )

    return RunSpec(
        name=f"med_dialog,subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MedDialog"],
    )


@run_spec_function("med_mcqa")
def get_med_mcqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.med_mcqa_scenario.MedMCQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Give a letter answer among A, B, C or D.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="med_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["MedMCQA"],
    )


@run_spec_function("med_paragraph_simplification")
def get_med_paragraph_simplification_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_paragraph_simplification_scenario.MedParagraphSimplificationScenario",
        args={},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=10,
        max_tokens=512,
        temperature=0.3,
    )

    return RunSpec(
        name="med_paragraph_simplification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MedParagraphSimplification"],
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


@run_spec_function("pubmed_qa")
def get_pubmed_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.pubmed_qa_scenario.PubMedQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no or C for maybe.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="pubmed_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["pubmed_qa"],
    )
