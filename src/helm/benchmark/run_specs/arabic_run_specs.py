"""Run specs for Arabic leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec, get_generation_adapter_spec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_ARABIC_REFERENCE_PREFIX_CHARACTERS = ["أ", "ب", "ج", "د", "هـ"]
_ARABIC_OUTPUT_MAPPING_PATTERN = "(أ|ب|ج|د|هـ)"


@run_spec_function("arabic_mmlu")
def get_arabic_mmlu_spec(subset: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_mmlu_scenario.ArabicMMLUScenario", args={"subset": subset}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"arabic_mmlu:subset={subset}",
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
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"alghafa:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["alghafa", f"alghafa_{subset}"],
    )


@run_spec_function("aratrust")
def get_aratrust_spec(category: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.aratrust_scenario.AraTrustScenario",
        args={"category": category},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"aratrust:category={category}",
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
        stop_sequences=[],
    )

    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.alrage_annotator.ALRAGEAnnotator")]

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.alrage_metric.ALRAGEMetric")
    ] + get_basic_metric_specs([])

    return RunSpec(
        name="alrage",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["alrage"],
    )


@run_spec_function("madinah_qa")
def get_madinah_qa_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.madinah_qa_scenario.MadinahQAScenario", args={"subset": subset}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"madinah_qa:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["madinah_qa"],
    )


@run_spec_function("mbzuai_human_translated_arabic_mmlu")
def get_arabic_mmmlu_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mbzuai_human_translated_arabic_mmlu.MBZUAIHumanTranslatedArabicMMLUScenario",
        args={"subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"mbzuai_human_translated_arabic_mmlu:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mbzuai_human_translated_arabic_mmlu"],
    )


@run_spec_function("arabic_exams")
def get_arabic_exams_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_exams_scenario.ArabicEXAMSScenario",
        args={"subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=100,
        reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
        output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
    )

    return RunSpec(
        name=f"arabic_exams:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_exams"],
    )
