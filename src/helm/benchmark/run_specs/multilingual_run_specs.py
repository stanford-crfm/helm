from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("mmmlu")
def get_mmmlu_spec(locale: str, subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmmlu_scenario.MMMLUScenario", args={"locale": locale, "subject": subject}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}. Answer the last question. Respond only with only a single letter corresponding to your choice.",  # noqa: E501
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"mmmlu:locale={locale},subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmmlu", f"mmmlu_{locale}_{subject}"],
    )


@run_spec_function("exams_multilingual")
def get_exams_multilingual_spec(language: str, subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.exams_multilingual_scenario.EXAMSMultilingualScenario",
        args={"language": language, "subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}. Answer the last question. Respond only with only a single letter corresponding to your choice.",  # noqa: E501
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"exams_multilingual:locale={language},subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["exams_multilingual", f"exams_multilingual_{language}_{subject}"],
    )
