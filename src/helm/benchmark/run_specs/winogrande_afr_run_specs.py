"""Run spec functions for Winogrande human-translated into 11 African languages

Available langs: "af", "zu", "xh", "am", "bm", "ig", "nso", "sn", "st", "tn", "ts" (see lang_map below for language code mapping to language name, or here for ISO code reference: https://huggingface.co/languages)
"""  # noqa: E501

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("winogrande_afr")
def get_winogrande_afr_spec(lang: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.winogrande_afr_scenario.Winogrande_Afr_Scenario", args={"lang": lang}
    )

    lang_map = {
        "af": "Afrikaans",
        "zu": "Zulu",
        "xh": "Xhosa",
        "am": "Amharic",
        "bm": "Bambara",
        "ig": "Igbo",
        "nso": "Sepedi",
        "sn": "Shona",
        "st": "Sesotho",
        "tn": "Setswana",
        "ts": "Tsonga",
    }

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are binary choice fill-in-the-blank sentences (with answers), "
        f"requiring common sense reasoning in {lang_map[lang]}.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"winogrande_afr:lang={lang},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["winogrande_afr", f"winogrande_afr_{lang}"],
    )
