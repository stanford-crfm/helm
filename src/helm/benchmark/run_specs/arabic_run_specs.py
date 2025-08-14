"""Run specs for Arabic leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec, get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_ARABIC_REFERENCE_PREFIX_CHARACTERS = ["أ", "ب", "ج", "د", "هـ"]


@run_spec_function("arabic_mmlu")
def get_arabic_mmlu_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.arabic_mmlu_scenario.ArabicMMLUScenario")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
    )

    return RunSpec(
        name="arabic_mmlu",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_mmlu"],
    )


@run_spec_function("alghafa")
def get_alghafa_spec(subset: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.alghafa_scenario.AlGhafaScenario", args={"subset": subset}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
    )

    return RunSpec(
        name=f"alghafa:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["alghafa", f"alghafa_{subset}"],
    )


@run_spec_function("aratrust")
def get_aratrust_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.aratrust_scenario.AraTrustScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب أو ج",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
    )

    return RunSpec(
        name="aratrust",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["aratrust"],
    )


@run_spec_function("alrage")
def get_alrage_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.alrage_scenario.ALRAGEScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
    )

    return RunSpec(
        name="alrage",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["alrage"],
    )


@run_spec_function("madinah_qa")
def get_madinah_qa_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.madinah_qa_scenario.MadinahQAScenario")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
    )

    return RunSpec(
        name="madinah_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["madinah_qa"],
    )


@run_spec_function("arabic_mmmlu")
def get_arabic_mmmlu_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmmlu_scenario.MMMLUScenario", args={"locale": "AR_XY", "subject": subject}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
    )

    return RunSpec(
        name=f"arabic_mmmlu:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_mmmlu", f"arabic_mmmlu_{subject}"],
    )


@run_spec_function("arabic_exams")
def get_arabic_exams_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.exams_multilingual_scenario.EXAMSMultilingualScenario",
        args={"language": "Arabic", "subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
    )

    return RunSpec(
        name=f"arabic_exams:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_exams", f"arabic_exams_{subject}"],
    )
