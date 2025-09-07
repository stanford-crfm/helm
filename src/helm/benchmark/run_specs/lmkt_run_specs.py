"""Run spec functions for Vietnam WVS cultural alignment evaluation."""

from typing import Optional, Dict, Literal
from helm.benchmark.adaptation.adapter_spec import (
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_open_ended_generation_metric_specs,
    get_regression_metric_specs,
)
from helm.benchmark.metrics.lmkt_metric_specs import get_semantic_similarity_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec


INSTRUCTIONS = {
    "cultural_value_understanding_wvs": {
        "en": {
            "instructions": "Please respond as the {country} persona described below.",
            "input_noun": "Question",
            "output_noun": "Answer",
        },
        "vi": {
            "instructions": "Vui lòng trả lời như một người {country} được mô tả bên dưới.",
            "input_noun": "Câu hỏi",
            "output_noun": "Trả lời",
        },
    },
    "social_norm_application_normad": {
        "en": {
            "instructions": "Please respond as the {country} persona described below.",
            "input_noun": "Situation",
            "output_noun": "Response",
        },
        "vi": {
            "instructions": "Vui lòng trả lời như một người {country} được mô tả bên dưới.",
            "input_noun": "Tình huống",
            "output_noun": "Phản hồi",
        },
    },
    "social_norm_reasoning_normad": {
        "en": {
            "instructions": "Please respond as the {country} persona described below.",
            "input_noun": "Situation",
            "output_noun": "Explanation",
        },
        "vi": {
            "instructions": "Vui lòng trả lời như một người {country} được mô tả bên dưới.",
            "input_noun": "Tình huống",
            "output_noun": "Giải thích",
        },
    },
    "cultural_evolution_understanding_culturebank": {
        "en": {
            "instructions": "Answer the quesstion in the below situation.",
            "input_noun": "Situation",
            "output_noun": "Answer",
        },
    },
}

COUNTRIES = {
    "US": "United States",
    "VN": "Vietnam",
}


@run_spec_function("cultural_value_understanding_wvs")
def get_cultural_value_understanding_wvs_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_scenarios.CulturalValueUnderstandingWVSScenario",
        args={
            "language": language,
            "num_personas": 300,
            "num_question_variants": 4,
            "include_few_shot_examples": True,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["cultural_value_understanding_wvs"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["cultural_value_understanding_wvs"][language]["input_noun"],
        output_noun=INSTRUCTIONS["cultural_value_understanding_wvs"][language]["output_noun"],
        max_tokens=3,
        stop_sequences=[],
    )

    return RunSpec(
        name=f"cultural_value_understanding_wvs:language={language},country={country}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["lmkt", "cultural_value_understanding_wvs"],
    )


@run_spec_function("social_norm_application_normad")
def get_social_norm_application_normad_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_scenarios.SocialNormApplicationNormADScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["social_norm_application_normad"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["social_norm_application_normad"][language]["input_noun"],
        output_noun=INSTRUCTIONS["social_norm_application_normad"][language]["output_noun"],
        max_tokens=5,
        stop_sequences=[],
    )

    return RunSpec(
        name=f"social_norm_application_normad:language={language},country={country}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["lmkt", "social_norm_application_normad"],
    )


@run_spec_function("social_norm_reasoning_normad")
def get_social_norm_reasoning_normad_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_scenarios.SocialNormReasoningNormADScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["social_norm_reasoning_normad"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["social_norm_reasoning_normad"][language]["input_noun"],
        output_noun=INSTRUCTIONS["social_norm_reasoning_normad"][language]["output_noun"],
        max_tokens=128,
        stop_sequences=[],
    )

    return RunSpec(
        name=f"social_norm_reasoning_normad:language={language},country={country}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_semantic_similarity_metric_specs(),
        groups=["lmkt", "social_norm_reasoning_normad"],
    )


@run_spec_function("cultural_knowledge_remembering_eclektic")
def get_cultural_knowledge_remembering_eclektic_spec(
    annotator_model: Optional[str] = "google/gemini-2.5-pro",
    annotator_model_deployment: Optional[str] = "google/gemini-2.5-pro",
) -> RunSpec:

    model: str = annotator_model or "google/gemini-2.5-pro"
    deployment: str = annotator_model_deployment or model

    annotator_args: Dict[str, str] = {
        "model": model,
        "model_deployment": deployment,
    }

    run_spec_name = (
        "cultural_knowledge_remembering_eclektic:" + f"annotator_model={annotator_args['model']}"
        f",annotator_model_deployment={annotator_args['model_deployment']}"
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_eclektic_scenario.CulturalKnowledgeRememberingEclekticScenario",
    )

    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        num_outputs=1,
        max_tokens=50,
        temperature=0.0,
    )

    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.lmkt_eclektic_annotator.EclekticAnnotator", args=annotator_args
        )
    ]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.lmkt_eclektic_metrics.EclekticMetric"),
    ]

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lmkt", "cultural_knowledge_remembering_eclektic"],
    )


@run_spec_function("cultural_safety_application_polyguard")
def get_cultural_safety_application_polyguard_spec(
    language: Optional[str],
    request_type: Literal["harmful", "unharmful", "both"] = "both",
    annotator_model: Optional[str] = "toxicityprompts/polyguard-qwen-smol",
    annotator_model_deployment: Optional[str] = "huggingface/polyguard-qwen-smol",
) -> RunSpec:

    model: str = annotator_model or "toxicityprompts/polyguard-qwen-smol"
    deployment: str = annotator_model_deployment or model

    annotator_args: Dict[str, str] = {
        "model": model,
        "model_deployment": deployment,
    }
    run_spec_name = (
        "cultural_safety_application_polyguard:"
        f"language={language},request_type={request_type},"
        f"annotator_model={annotator_args['model']},"
        f"annotator_model_deployment={annotator_args['model_deployment']}"
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_polyguard_scenario.CulturalSafetyApplicationPolyGuardScenario",
        args={"language": language, "request_type": request_type},
    )

    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        num_outputs=1,
        max_tokens=50,
        temperature=0.0,
    )

    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.lmkt_polyguard_annotator.PolyGuardAnnotator", args=annotator_args
        )
    ]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.lmkt_polyguard_metrics.PolyGuardMetric"),
    ]

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lmkt", "cultural_safety_application_polyguard"],
    )


@run_spec_function("cultural_evolution_understanding_culturebank")
def get_cultural_evolution_understanding_culturebank_spec(language: str, datasplit: str = "reddit") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_scenarios.CulturalEvolutionUnderstandingCultureBankScenario",
        args={
            "language": language,
            "datasplit": datasplit,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["cultural_evolution_understanding_culturebank"][language]["instructions"],
        input_noun=INSTRUCTIONS["cultural_evolution_understanding_culturebank"][language]["input_noun"],
        output_noun=INSTRUCTIONS["cultural_evolution_understanding_culturebank"][language]["output_noun"],
        max_tokens=5,
        stop_sequences=[],
    )

    return RunSpec(
        name=f"cultural_evolution_understanding_culturebank:language={language},datasplit={datasplit}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_regression_metric_specs(),
        groups=["lmkt", "cultural_evolution_understanding_culturebank"],
    )
