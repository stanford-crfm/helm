"""Run spec functions for Winogrande human-translated into 11 African languages

Available langs: "af", "zu", "xh", "am", "bm", "ig", "nso", "sn", "st", "tn", "ts" (see lang_map below for language code mapping to language name, or here for ISO code reference: https://huggingface.co/languages)
"""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_machine_translation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


@run_spec_function("winogrande_afr")
def get_winogrande_afr_spec(lang: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.winogrande_afr_scenario.Winogrande_Afr_Scenario", args={"lang": lang}
    )

    lang_map = {
        'af': 'Afrikaans',
        'zu': 'Zulu',
        'xh': 'Xhosa',
        'am': 'Amharic',
        'bm': 'Bambara',
        'ig': 'Igbo',
        'nso': 'Sepedi',
        'sn': 'Shona',
        'st': 'Sesotho',
        'tn': 'Setswana',
        'ts': 'Tsonga',
    }

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are binary choice fill-in-the-blank sentences (with answers), requiring common sense reasoning "
                     f"in {lang_map[lang]}.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"winogrande_afr:lang={lang},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["low_resource_languages"],
    )
