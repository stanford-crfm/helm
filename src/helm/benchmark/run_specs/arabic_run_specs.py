"""Run specs for Arabic leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec, get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arabic_mmlu")
def get_arabic_mmlu_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.arabic_mmlu_scenario.ArabicMMLUScenario")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
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
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب أو ج.",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
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
        instructions="Read the passaage, then answer the following question.",  # noqa: E501
        input_noun="",
        output_noun="Answer",
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
        instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة الرد بحرف واحد فقط",  # noqa: E501
        # instructions="السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة",  # noqa: E501
        input_noun="السؤال",
        output_noun="الإجابة",
        max_tokens=1000,
    )

    return RunSpec(
        name="madinah_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["madinah_qa"],
    )
