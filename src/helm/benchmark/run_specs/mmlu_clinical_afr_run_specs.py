"""Run spec functions for three clinical sections of MMLU human-translated into 11 African languages

Available subjects: "clinical_knowledge", "college_medicine", "virology"
Available langs: "af", "zu", "xh", "am", "bm", "ig", "nso", "sn", "st", "tn", "ts" (see lang_map below for language code mapping to language name, or here for ISO code reference: https://huggingface.co/languages)
"""  # noqa: E501

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("mmlu_clinical_afr")
def get_mmlu_clinical_afr_spec(subject: str, lang: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_clinical_afr_scenario.MMLU_Clinical_Afr_Scenario",
        args={"subject": subject, "lang": lang},
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
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')} "
        f"in {lang_map[lang]}.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"mmlu_clinical_afr:subject={subject},lang={lang},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[f"mmlu_clinical_afr_{subject}", f"mmlu_clinical_afr_{subject}_{lang}"],
    )
