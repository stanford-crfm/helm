"""Run spec functions for CLEVA.

CLEVA is a benchmark for holistically evaluating Chinese LLMs.

Paper: https://arxiv.org/abs/2308.04813"""

import itertools
from functools import partial
from typing import Callable, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_language_modeling_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_generic_metric_specs,
    get_language_modeling_metric_specs,
    get_multiple_choice_classification_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


def get_cleva_machine_translation_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.machine_translation_metrics.CLEVAMachineTranslationMetric", args={}
        )
    ] + get_basic_metric_specs([])


def get_cleva_paraphrase_generation_metric_specs(alpha: float = 0.8) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.paraphrase_generation_metrics.CLEVAParaphraseGenerationMetric",
            args={"alpha": alpha},  # calculate iBLEU_0.8 by default
        )
    ] + get_basic_metric_specs([])


def get_cleva_topk_accuracy_metric_specs(k: int = 1, cut_off: int = 5) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_accuracy_metrics.CLEVATopKAccuracyMetric",
            args={"k": k, "cut_off": cut_off},
        )
    ]


def get_cleva_bias_metric_specs() -> List[MetricSpec]:
    demographic_categories = ["race", "gender"]
    target_categories = ["adjective", "profession"]
    cross_dem_target = itertools.product(demographic_categories, target_categories)

    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVABiasMetric",
            args={"mode": "associations", "demographic_category": dem, "target_category": tgt},
        )
        for dem, tgt in cross_dem_target
    ] + [
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVABiasMetric",
            args={"mode": "representation", "demographic_category": dem},
        )
        for dem in demographic_categories
    ]


def get_cleva_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVAToxicityMetric", args={}),
    ]


def get_cleva_generative_harms_metric_specs(include_basic_metrics: bool = False) -> List[MetricSpec]:
    return (
        get_cleva_bias_metric_specs()
        + get_cleva_toxicity_metric_specs()
        + (get_basic_metric_specs([]) if include_basic_metrics else [])
    )


def get_cleva_copyright_metric_spec(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVACopyrightMetric",
            args={**args, "name": "longest_common_prefix_length"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVACopyrightMetric",
            args={**args, "name": "edit_distance"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.cleva_harms_metrics.CLEVACopyrightMetric",
            args={**args, "name": "edit_similarity"},
        ),
    ]


def get_cleva_generative_task_metric_spec(task: str, subtask: Optional[str], **kwargs) -> List[MetricSpec]:
    CLEVA_GEN_TASK_TO_METRIC: Dict[str, Callable] = {
        "opinion_mining:opinion_target_extraction": get_exact_match_metric_specs,
        "paraphrase_generation": get_cleva_paraphrase_generation_metric_specs,
        "closed_book_question_answering:generative_question_answering": get_exact_match_metric_specs,
        "conceptual_generalization": get_cleva_topk_accuracy_metric_specs,
        "translation:en2zh": get_cleva_machine_translation_metric_specs,
        "translation:zh2en": get_cleva_machine_translation_metric_specs,
        "mathematical_calculation:add": get_exact_match_metric_specs,
        "mathematical_calculation:sub": get_exact_match_metric_specs,
        "mathematical_calculation:mul": get_exact_match_metric_specs,
        "inductive_reasoning:add": get_exact_match_metric_specs,
        "inductive_reasoning:sub": get_exact_match_metric_specs,
        "inductive_reasoning:mul": get_exact_match_metric_specs,
        "reasoning_primitive:dyck_language": get_exact_match_metric_specs,
        "reasoning_primitive:pattern_induction": get_exact_match_metric_specs,
        "reasoning_primitive:pattern_matching": get_exact_match_metric_specs,
        "reasoning_primitive:variable_sub": get_exact_match_metric_specs,
        "subject_knowledge:art": get_exact_match_metric_specs,
        "subject_knowledge:biomedicine": get_exact_match_metric_specs,
        "subject_knowledge:chemistry": get_exact_match_metric_specs,
        "subject_knowledge:computer_science": get_exact_match_metric_specs,
        "subject_knowledge:economics": get_exact_match_metric_specs,
        "subject_knowledge:geography": get_exact_match_metric_specs,
        "subject_knowledge:history": get_exact_match_metric_specs,
        "subject_knowledge:law": get_exact_match_metric_specs,
        "subject_knowledge:literature": get_exact_match_metric_specs,
        "subject_knowledge:math": get_exact_match_metric_specs,
        "subject_knowledge:other_general": get_exact_match_metric_specs,
        "subject_knowledge:philosophy": get_exact_match_metric_specs,
        "subject_knowledge:physics": get_exact_match_metric_specs,
        "subject_knowledge:politics": get_exact_match_metric_specs,
        "summarization:dialogue_summarization": partial(get_basic_metric_specs, ["chinese_rouge_2"]),
        "pinyin_transliteration:pinyin2zh": partial(get_basic_metric_specs, ["chinese_bleu_1"]),
        "pinyin_transliteration:zh2pinyin": partial(get_basic_metric_specs, ["chinese_bleu_1"]),
        "dialogue_generation:task_oriented": partial(get_basic_metric_specs, ["chinese_bleu_1"]),
        "data_to_text_generation": partial(get_basic_metric_specs, ["chinese_bleu_1"]),
        "mathematical_reasoning:math_word_problem": partial(get_basic_metric_specs, ["cleva_math_result_match"]),
    }

    key: str = task
    if subtask is not None:
        key += ":" + subtask
    return CLEVA_GEN_TASK_TO_METRIC[key](**kwargs)


@run_spec_function("cleva")
def get_cleva_spec(task: str, version: str, subtask: Optional[str] = None, prompt_id: int = 0) -> RunSpec:
    from helm.benchmark.scenarios.cleva_scenario import CLEVAScenario  # noqa

    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), CLEVAScenario.name)
    CLEVAScenario.download_dataset(task, version, scenario_cache_path)

    _, prompt_setting = CLEVAScenario.get_prompt_setting(task, subtask, version, prompt_id, scenario_cache_path)
    inference_parameters = CLEVAScenario.load_inference_parameters(
        task, subtask, version, prompt_id, scenario_cache_path
    )

    class_name_prefix = "".join([word.capitalize() for word in task.split("_")])
    scenario_spec = ScenarioSpec(
        class_name=f"helm.benchmark.scenarios.cleva_scenario.CLEVA{class_name_prefix}Scenario",
        args={"version": version, "subtask": subtask, "prompt_id": prompt_id},
    )
    run_spec_name: str = f"cleva:task={task},version={version},prompt_id={prompt_id}"
    if subtask:
        run_spec_name += f",subtask={subtask}"

    if task in ["copyright"]:
        adapter_spec = get_completion_adapter_spec(
            temperature=inference_parameters.get("temperature", 0.2),
            max_tokens=inference_parameters.get("max_tokens", 1024),
            num_outputs=inference_parameters.get("num_outputs", 1),
        )
        args = {"normalize_by_prefix_length": True, "normalize_newline_space_tab": False}
        metric_specs = get_cleva_copyright_metric_spec(args) + get_cleva_generative_harms_metric_specs()
    elif task in ["code_synthesis"]:
        adapter_spec = get_completion_adapter_spec(
            instructions=prompt_setting.instructions,
            temperature=inference_parameters.get("temperature", 0.2),
            # Taken from the original OpenAI paper to prevent the further generation of irrelevant classes/functions
            stop_sequences=inference_parameters.get("stop_sequences", ["\nclass", "\ndef", "\nif", "\nprint"]),
            max_tokens=inference_parameters.get("max_tokens", 600),
        )
        metric_specs = (
            get_basic_generation_metric_specs(["code_eval_acc", "pass"])
            + get_generic_metric_specs()
            + get_cleva_generative_harms_metric_specs()
        )
    elif task in ["language_modeling"]:
        adapter_spec = get_language_modeling_adapter_spec()
        metric_specs = get_language_modeling_metric_specs([])
    else:
        if prompt_setting.method in [
            ADAPT_MULTIPLE_CHOICE_JOINT,
            ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
            ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        ]:
            if prompt_setting.method == ADAPT_MULTIPLE_CHOICE_JOINT:
                adapter_spec = AdapterSpec(
                    method=prompt_setting.method,
                    instructions=prompt_setting.instructions,
                    input_prefix=prompt_setting.input_prefix,
                    input_suffix=prompt_setting.input_suffix,
                    output_prefix=prompt_setting.output_prefix,
                    output_suffix=prompt_setting.output_suffix,
                    max_train_instances=inference_parameters.get("max_train_instances", 5),
                    num_outputs=inference_parameters.get("num_outputs", 5),
                    max_tokens=inference_parameters.get("max_tokens", 1),
                    temperature=inference_parameters.get("temperature", 0.0),
                    stop_sequences=inference_parameters.get("stop_sequences", ["\n"]),
                    sample_train=inference_parameters.get("sample_train", True),
                    multi_label=inference_parameters.get("multi_label", False),
                )
            else:
                adapter_spec = AdapterSpec(
                    method=prompt_setting.method,
                    instructions=prompt_setting.instructions,
                    input_prefix=prompt_setting.input_prefix,
                    input_suffix=prompt_setting.input_suffix,
                    output_prefix=prompt_setting.output_prefix,
                    output_suffix=prompt_setting.output_suffix,
                    # Separate is basically language modeling, so can't easily use in-context examples
                    max_train_instances=inference_parameters.get("max_train_instances", 5),
                    num_outputs=1,
                    max_tokens=0,
                    temperature=inference_parameters.get("temperature", 0.0),
                    sample_train=inference_parameters.get("sample_train", True),
                )
            metric_specs = get_exact_match_metric_specs()
            if task in ["fact_checking", "bias"]:
                metric_specs += get_multiple_choice_classification_metric_specs()
        elif prompt_setting.method == ADAPT_GENERATION:
            adapter_spec = AdapterSpec(
                method=prompt_setting.method,
                instructions=prompt_setting.instructions,
                input_prefix=prompt_setting.input_prefix,
                input_suffix=prompt_setting.input_suffix,
                output_prefix=prompt_setting.output_prefix,
                output_suffix=prompt_setting.output_suffix,
                max_train_instances=inference_parameters.get("max_train_instances", 5),
                num_outputs=inference_parameters.get("num_outputs", 1),
                max_tokens=inference_parameters.get("max_tokens", 20),
                temperature=inference_parameters.get("temperature", 0.0),
                stop_sequences=inference_parameters.get("stop_sequences", ["\n"]),
                sample_train=inference_parameters.get("sample_train", True),
                multi_label=inference_parameters.get("multi_label", True),
            )
            metric_specs = (
                get_cleva_generative_task_metric_spec(task, subtask) + get_cleva_generative_harms_metric_specs()
            )
        else:
            raise ValueError(
                f"{task} can only be {ADAPT_GENERATION}, {ADAPT_MULTIPLE_CHOICE_JOINT}, "
                f"{ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED} or {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL}"
            )

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["cleva", f"cleva_{task}"],
    )
