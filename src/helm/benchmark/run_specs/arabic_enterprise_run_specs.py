"""Run specs for Arabic Enterprise leaderboard

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


_ARABIC_REFERENCE_PREFIX_CHARACTERS = ["أ", "ب", "ج"]
_ARABIC_OUTPUT_MAPPING_PATTERN = "(أ|ب|ج)"


@run_spec_function("arabic_content_generation")
def get_arabic_content_generation_spec(category: str, annotator_type: str = "absolute") -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_content_generation_scenario.ArabicContentGenerationScenario",
        args={"category": category},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Given the following lists of facts and style features, generate a professional business article in Modern Standard Arabic. Use all of the provided facts. Use only the provided facts, and do not introduce new facts. Respond only with the article.",
        max_tokens=1000,
        stop_sequences=[],
    )

    if annotator_type == "absolute":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_annotator.ArabicContentGenerationAnnotator"
            )
        ]
    elif annotator_type == "relative":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_relative_annotator.ArabicContentGenerationRelativeAnnotator"
            )
        ]
    elif annotator_type == "similarity":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_similarity_annotator.ArabicContentGenerationSimilarityAnnotator"
            )
        ]
    else:
        raise ValueError(f"Unknown annotator_type {annotator_type}")
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_content_generation_metric.ArabicContentGenerationMetric")
    ]

    return RunSpec(
        name=f"arabic_content_generation:category={category},annotator_type={annotator_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["arabic_content_generation"],
    )


@run_spec_function("arabic_finance_mcq")
def get_arabic_finance_mcq_spec(lang: str = "ar") -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_finance_scenario.ArabicFinanceMCQScenario", args={"lang": lang}
    )

    if lang == "en":
        adapter_spec = get_multiple_choice_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            instructions="Answer the following multiple-choice question with only a single letter.",  # noqa: E501
            input_noun="Question",
            output_noun="Answer",
            max_tokens=10,
        )
    elif lang == "ar":
        adapter_spec = get_multiple_choice_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            instructions="السؤال التالي هو سؤال متعدد الاختيارات. اختر الإجابة الصحيحة اكتب حرف الإجابة فقط، دون أي إضافات أخرى.",  # noqa: E501
            input_noun="السؤال",
            output_noun="الإجابة",
            max_tokens=10,
            reference_prefix_characters=_ARABIC_REFERENCE_PREFIX_CHARACTERS,
            output_mapping_pattern=_ARABIC_OUTPUT_MAPPING_PATTERN,
        )
    else:
        raise ValueError(f"Unknown value for `lang`: {lang}")

    if lang == "ar":
        run_spec_name = "arabic_finance_mcq"
    else:
        run_spec_name = f"arabic_finance_mcq:lang={lang}"

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_finance_mcq"],
    )


@run_spec_function("arabic_finance_bool")
def get_arabic_finance_bool_spec(lang: str = "ar") -> RunSpec:
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

    if lang == "ar":
        run_spec_name = "arabic_finance_bool"
    else:
        run_spec_name = f"arabic_finance_bool:lang={lang}"

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_finance_bool"],
    )


@run_spec_function("arabic_finance_calculation")
def get_arabic_finance_calculation_spec(lang: str = "ar") -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_finance_scenario.ArabicFinanceCalculationScenario",
        args={"lang": lang},
    )

    if lang == "en":
        instructions = (
            "Answer the question, giving your reasoning beforehand. Wrap the final answer with the \\boxed{} command."
        )
    elif lang == "ar":
        instructions = "أجب عن السؤال، مع تقديم تعليلك مسبقًا. ضع الإجابة النهائية داخل الأمر \\boxed{}."
    else:
        raise ValueError(f"Unknown value for `lang`: {lang}")

    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
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
    if lang == "ar":
        run_spec_name = "arabic_finance_calculation"
    else:
        run_spec_name = f"arabic_finance_calculation:lang={lang}"

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["arabic_finance_calculation"],
    )


@run_spec_function("arabic_legal_qa")
def get_arabic_legal_qa() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.arabic_legal_scenario.ArabicLegalQAScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the following question briefly in Modern Standard Arabic in the context of the UAE. Respond only with the answer.",
        max_tokens=1000,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.arabic_legal_annotator.ArabicLegalAnnotator")
    ]

    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_legal_metric.ArabicLegalMetric")
    ]

    return RunSpec(
        name="arabic_legal_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["arabic_legal_qa"],
    )


@run_spec_function("arabic_legal_rag")
def get_arabic_legal_rag() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.arabic_legal_scenario.ArabicLegalRAGScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the following question briefly in Modern Standard Arabic in the context of the UAE. Respond only with the answer.",
        max_tokens=1000,
        stop_sequences=[],
    )

    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.arabic_legal_annotator.ArabicLegalAnnotator")
    ]

    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.arabic_legal_metric.ArabicLegalMetric")
    ]

    return RunSpec(
        name="arabic_legal_rag",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=["arabic_legal_rag"],
    )
