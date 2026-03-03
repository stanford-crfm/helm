"""Run specs for Arabic leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_ARABIC_REFERENCE_PREFIX_CHARACTERS = ["أ", "ب", "ج", "د", "هـ"]
_ARABIC_OUTPUT_MAPPING_PATTERN = "(أ|ب|ج|د|هـ)"


@run_spec_function("arabic_tool_usage")
def get_arabic_tool_usage_spec(lang: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_tool_usage_scenario.ArabicToolUsageScenario", args={"lang": lang}
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=100,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.arabic_tool_usage_annotator.ArabicToolUsageAnnotator")
    ]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_tool_usage_metric.ArabicToolUsageMetric")
    ]

    return RunSpec(
        name=f"arabic_tool_usage:lang={lang}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["arabic_tool_usage"],
    )


@run_spec_function("arabic_finance_mcq")
def get_arabic_finance_mcq_spec(lang: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_finance_scenario.ArabicFinanceMCQScenario", args={"lang": lang}
    )

    if lang == "en":

        adapter_spec = get_multiple_choice_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            instructions="Answer the following multiple-choice question with only a single letter.",  # noqa: E501
            input_noun="السؤال",
            output_noun="الإجابة",
            max_tokens=100,
            reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
            output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
        )
    elif lang == "ar":
        adapter_spec = get_multiple_choice_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            instructions="السؤال التالي هو سؤال متعدد الاختيارات. اختر الإجابة الصحيحة اكتب حرف الإجابة فقط، دون أي إضافات أخرى.",  # noqa: E501
            input_noun="السؤال",
            output_noun="الإجابة",
            max_tokens=100,
            reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
            output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
        )
    else:
        raise ValueError(f"Unknown value for `lang`: {lang}")

    return RunSpec(
        name=f"arabic_finance_mcq:lang={lang}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_finance_mcq"],
    )


@run_spec_function("arabic_finance_bool")
def get_arabic_finance_bool_spec(lang: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_finance_scenario.ArabicFinanceBoolScenario", args={"lang": lang}
    )

    if lang == "en":
        instructions = 'Is the following passage true? Answer only with "Yes" or "No".'
    elif lang == "ar":
        instructions = 'هل المقطع التالي صحيح؟ أجب فقط بـ "نعم" أو "لا".'
    else:
        raise ValueError(f"Unknown value for `lang`: {lang}")

    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
        max_tokens=10,
        stop_sequences=[],
    )

    return RunSpec(
        name=f"arabic_finance_bool:lang={lang}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_finance_bool"],
    )


@run_spec_function("arabic_finance_calculation")
def get_arabic_finance_calculation_spec(lang: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_finance_scenario.ArabicFinanceCalculationScenario",
        args={"lang": lang},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the question, giving your reasoning beforehand. Wrap the final answer with the \\boxed{} command.",
        max_tokens=2000,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.arabic_finance_calculation_annotator.ArabicFinanceCalculationAnnotator"
        )
    ]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_finance_calculation_metric.ArabicFinanceCalculationMetric")
    ]

    return RunSpec(
        name=f"arabic_finance_calculation:lang={lang}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["arabic_finance_calculation"],
    )


@run_spec_function("arabic_writing_style")
def get_arabic_writing_style_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_writing_style_scenario.ArabicWritingStyleScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=1000,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.arabic_writing_style_annotator.ArabicWritingStyleAnnotator")
    ]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_writing_style_metric.ArabicWritingStyleMetric")
    ]

    return RunSpec(
        name="arabic_writing_style",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["arabic_writing_style"],
    )
