"""Run spec functions for Vietnam WVS cultural alignment evaluation."""

from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.metrics.lmkt_metric_specs import get_semantic_similarity_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

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
    "social_norm_explanation_normad": {
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
        name="cultural_value_understanding_wvs",
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
        name="social_norm_application_normad",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["lmkt", "social_norm_application_normad"],
    )


@run_spec_function("social_norm_explanation_normad")
def get_social_norm_explanation_normad_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lmkt_scenarios.SocialNormExplanationNormADScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["social_norm_explanation_normad"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["social_norm_explanation_normad"][language]["input_noun"],
        output_noun=INSTRUCTIONS["social_norm_explanation_normad"][language]["output_noun"],
        max_tokens=128,
        stop_sequences=[],
    )

    return RunSpec(
        name="social_norm_explanation_normad",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_semantic_similarity_metric_specs(),
        groups=["lmkt", "social_norm_explanation_normad"],
    )
