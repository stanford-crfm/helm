from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_separate_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

# BHASA Run Specs
#   D. Linguistic Diagnostics

# D. Linguistic Diagnostics (LINDSEA)
#   1. Syntax
#   2. Pragmatics

# 1. Syntax: LINDSEA Minimal Pairs
LINDSEA_OUTPUT_NOUNS = {
    "id": "Jawaban"
}

@run_spec_function("lindsea_syntax_minimal_pairs")
def get_lindsea_syntax_minimal_pairs_spec(language: str = "id", method: str = "mcq") -> RunSpec:
    name = f"lindsea_syntax_minimal_pairs_{language}"
    if method == "mcq":
        adapter_spec = get_generation_adapter_spec(
            output_noun=LINDSEA_OUTPUT_NOUNS[language],
            max_tokens=2
        )
    else:
        adapter_spec = get_multiple_choice_separate_adapter_spec(
            method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
            empty_input=True,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEASyntaxMinimalPairsScenario",
        args={
            "method": method,
            "language": language,
        }
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_linguistic", f"lindsea_syntax_minimal_pairs_{language}"],
    )

# 2.1. Pragmatics: LINDSEA Pragmatic Reasoning (single sentence)
@run_spec_function("lindsea_pragmatics_pragmatic_reasoning_single")
def get_lindsea_pragmatics_pragmatic_reasoning_single_spec(language="id") -> RunSpec:
    name = f"lindsea_pragmatics_pragmatic_reasoning_single_{language}"

    adapter_spec = get_generation_adapter_spec(
        output_noun=LINDSEA_OUTPUT_NOUNS[language],
        stop_sequences=["\n"],
        max_train_instances=0,
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEAPragmaticsPragmaticReasoningSingleScenario",
        args={
            "language": language,
        }
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_linguistic", f"lindsea_pragmatics_pragmatic_reasoning_single_{language}"],
    )

# 2.2. Pragmatics: LINDSEA Pragmatic Reasoning (sentence pair)
@run_spec_function("lindsea_pragmatics_pragmatic_reasoning_pair")
def get_lindsea_pragmatics_pragmatic_reasoning_pair_spec(language="id") -> RunSpec:
    name = f"lindsea_pragmatics_pragmatic_reasoning_pair_{language}"

    adapter_spec = get_generation_adapter_spec(
        output_noun=LINDSEA_OUTPUT_NOUNS[language],
        stop_sequences=["\n"],
        max_train_instances=0,
        max_tokens=8,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEAPragmaticsPragmaticReasoningPairScenario",
        args={
            "language": language,
        }
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_linguistic", f"lindsea_pragmatics_pragmatic_reasoning_pair_{language}"],
    )