import importlib
import itertools
from functools import partial
from typing import Any, Callable, List, Dict, Optional, Set, TypeVar

from helm.common.hierarchical_logger import hlog, htrack
from helm.common.object_spec import ObjectSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_GENERATION,
    ADAPT_RANKING_BINARY,
)
from helm.benchmark.adaptation.adapters.binary_ranking_adapter import BinaryRankingAdapter
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.common.optional_dependencies import handle_module_not_found_error
from .metrics.metric import MetricSpec
from .run_expander import (
    RUN_EXPANDERS,
    RunExpander,
    GlobalPrefixRunExpander,
    StopRunExpander,
    ChatMLRunExpander,
    AddToStopRunExpander,
    IncreaseMaxTokensRunExpander,
    FormatPromptRunExpander,
    IncreaseTemperatureRunExpander,
)
from .runner import RunSpec
from .scenarios.lex_glue_scenario import (
    get_lex_glue_max_train_instances,
    get_lex_glue_instructions,
    get_lex_glue_max_tokens,
    get_lex_glue_task_type,
)
from .scenarios.scenario import ScenarioSpec
from .scenarios.big_bench_scenario import BIGBenchScenario
from .scenarios.msmarco_scenario import MSMARCOScenario
from .scenarios.copyright_scenario import datatag2hash_code
from .scenarios.raft_scenario import get_raft_instructions
from .scenarios.lextreme_scenario import (
    get_lextreme_instructions,
    get_lextreme_max_train_instances,
    get_lextreme_max_tokens,
    TaskType,
    get_lextreme_task_type,
)
from helm.proxy.models import (
    ANTHROPIC_CLAUDE_1_MODEL_TAG,
    ANTHROPIC_CLAUDE_2_MODEL_TAG,
    get_model,
    NO_NEWLINES_TAG,
    NLG_PREFIX_TAG,
    CHATML_MODEL_TAG,
    OPENAI_CHATGPT_MODEL_TAG,
    BUGGY_TEMP_0_TAG,
)
from helm.common.general import singleton


############################################################
# Prototypical adapter specs


def format_instructions(instructions: str) -> str:
    if len(instructions) > 0:
        instructions += "\n"
    return instructions


def get_multiple_choice_joint_adapter_spec(
    instructions: str,
    input_noun: Optional[str],
    output_noun: str,
    num_outputs: int = 5,
    max_train_instances: int = 5,
    max_tokens: int = 5,
    sample_train: bool = True,
    **kwargs,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]: [output]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]:
    """

    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=format_instructions(instructions),
        input_prefix=f"{input_noun}: " if input_noun is not None else "",
        input_suffix="\n" if input_noun is not None else "",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=["\n"],
        sample_train=sample_train,
        **kwargs,
    )


def get_multiple_choice_separate_adapter_spec(method: str, empty_input: bool = False) -> AdapterSpec:
    """
    [input] [reference_i]
    or
    [reference_i]
    """
    assert method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}

    return AdapterSpec(
        method=method,
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix=" " if not empty_input else "",
        output_suffix="",
        # Separate is basically language modeling, so can't easily use in-context examples
        max_train_instances=0,
        num_outputs=1,
        max_tokens=0,
        temperature=0.0,
    )


def get_multiple_choice_adapter_spec(
    method: str,
    instructions: str,
    input_noun: Optional[str],
    output_noun: str,
    max_train_instances: int = 5,
    num_outputs: int = 5,
    max_tokens: int = 1,
    empty_input: bool = False,
    sample_train: bool = True,
    **kwargs,
):
    """
    Toggle between joint and separate adapters.
    """
    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        return get_multiple_choice_joint_adapter_spec(
            instructions,
            input_noun,
            output_noun,
            max_train_instances=max_train_instances,
            num_outputs=num_outputs,
            max_tokens=max_tokens,
            sample_train=sample_train,
            **kwargs,
        )
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        return get_multiple_choice_separate_adapter_spec(method, empty_input)
    else:
        raise ValueError(f"Invalid adaptation method: {method}")


def get_ranking_binary_adapter_spec(
    instructions: str = "",
    document_noun: str = "Passage",
    query_noun: str = "Query",
    output_prefix: str = "Does the passage answer the query?",
    output_noun: str = "Answer",
    max_train_instances: int = 4,
    num_outputs: int = 1,
    num_train_trials: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 5,
    **kwargs,
) -> AdapterSpec:
    """
    [instructions]

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]

    ...

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]
    """
    msg = (
        "There must be an even number of in-context examples to ensure that"
        "an equal number of positive and negative examples are included."
    )
    assert max_train_instances % 2 == 0, msg
    max_train_instances = int(max_train_instances / 2)

    return AdapterSpec(
        method=ADAPT_RANKING_BINARY,
        instructions=format_instructions(instructions),
        input_prefix=f"{query_noun}: ",
        input_suffix="\n",
        reference_prefix=f"{document_noun}: ",
        reference_suffix="\n",
        output_prefix=f"{output_prefix}\n{output_noun}: ",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        num_train_trials=num_train_trials,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_completion_adapter_spec(
    instructions: str = "",
    input_prefix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_train_instances: int = 0,
    temperature: float = 0.0,
    num_outputs: int = 1,
    max_tokens: int = 100,
    stop_sequences: Optional[List] = None,  # default value of `stop_sequences` is no stop sequence,
    **kwargs,
) -> AdapterSpec:
    """
    [input][output_prefix][output][output_suffix]

    [input][output_prefix]
    """
    if stop_sequences is None:
        stop_sequences = []

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix,
        input_suffix="",
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        max_train_instances=max_train_instances,
        temperature=temperature,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        **kwargs,
    )


def get_generation_adapter_spec(
    instructions: str = "",
    input_noun: Optional[str] = None,
    newline_after_input_noun: bool = False,
    output_noun: Optional[str] = None,
    newline_after_output_noun: bool = False,
    max_train_instances: int = 5,
    num_outputs: int = 1,
    max_tokens: int = 5,
    stop_sequences: Optional[List] = None,  # default value of `stop_sequences` is ["\n"]
    temperature: float = 0.0,
    multi_label: bool = False,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [output_noun]: [output]

    [input_noun]: [input]
    [output_noun]:
    """

    def format_prefix(noun: Optional[str], append_new_line: bool) -> str:
        """
        When `append_new_line` is False:
            [input_noun]: [input]

        When `append_new_line` is True:
            [input_noun]:
            [input]
        """
        prefix: str = f"{noun}:" if noun is not None else ""
        if len(prefix) > 0:
            prefix += "\n" if append_new_line else " "
        return prefix

    if stop_sequences is None:
        stop_sequences = ["\n"]

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=format_instructions(instructions),
        input_prefix=format_prefix(input_noun, append_new_line=newline_after_input_noun),
        input_suffix="\n",
        output_prefix=format_prefix(output_noun, append_new_line=newline_after_output_noun),
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        multi_label=multi_label,
    )


def get_instruct_adapter_spec(
    num_outputs: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AdapterSpec:
    """
    Zero-shot instruction-following.
    """
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=[],
    )


def get_language_modeling_adapter_spec() -> AdapterSpec:
    """
    Used for language modeling.
    """
    return AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=0,
        temperature=0.0,
    )


def get_summarization_adapter_spec(num_sents: Optional[int], max_train_instances: int = 5, **kwargs) -> AdapterSpec:
    """
    Used for summarization.
    """

    if num_sents == 1:
        out_pref = "Summarize the above article in 1 sentence.\n"
    elif num_sents is None:
        out_pref = "Summarize the above article.\n"
    else:
        out_pref = f"Summarize the above article in {num_sents} sentences.\n"

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="###\nArticle: ",
        input_suffix="\n\n",
        output_prefix=out_pref,
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        stop_sequences=["###"],  # Separator between few-shot instances.
        **kwargs,
    )


def get_machine_translation_adapter_spec(
    source_language, target_language, max_train_instances, **kwargs
) -> AdapterSpec:
    """
    Used for machine translation.
    """
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=f"Translate {source_language} to {target_language}:",
        input_prefix="",
        input_suffix=" = ",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        stop_sequences=["\n\n"],
        temperature=0.0,
        **kwargs,
    )


############################################################
# Examples of scenario and adapter specs


def get_scenario_spec1() -> ScenarioSpec:
    return ScenarioSpec(
        class_name="helm.benchmark.scenarios.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )


def get_scenario_spec_tiny():
    return ScenarioSpec(
        class_name="helm.benchmark.scenarios.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 2, "num_test_instances": 2},
    )


def get_adapter_spec1() -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.\n",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )


############################################################
# Metrics


def get_basic_metric_specs(names: List[str]) -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={"names": names})]


def get_exact_match_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "prefix_exact_match", "quasi_prefix_exact_match"]
    )


def get_f1_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match", "f1_score"])


def get_classification_metric_specs(delimiter: Optional[str] = None) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.classification_metrics.ClassificationMetric",
            args={"delimiter": delimiter},
        )
    ]


def get_multiple_choice_classification_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={}
        )
    ]


def get_bbq_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bbq_metrics.BBQMetric", args={})
    ] + get_exact_match_metric_specs()


def get_msmarco_metric_specs(track: str, rank: Optional[int] = None) -> List[MetricSpec]:
    # Names of the measures we want to compute.
    measure_names = MSMARCOScenario.MEASURE_NAMES[track]
    multiple_relevance_values = set(MSMARCOScenario.GOLD_RELATIONS[track]) != {1}

    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.ranking_metrics.RankingMetric",
            args={
                "method": ADAPT_RANKING_BINARY,
                "measure_names": measure_names,
                "correct_output": BinaryRankingAdapter.RANKING_CORRECT_LABEL,
                "wrong_output": BinaryRankingAdapter.RANKING_WRONG_LABEL,
                "rank": rank,
                "multiple_relevance_values": multiple_relevance_values,
            },
        ),
    ] + get_basic_metric_specs(names=[])


def get_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.toxicity_metrics.ToxicityMetric", args={}),
    ]


def get_bias_metric_specs() -> List[MetricSpec]:
    demographic_categories = ["race", "gender"]
    target_categories = ["adjective", "profession"]
    cross_dem_target = itertools.product(demographic_categories, target_categories)

    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.bias_metrics.BiasMetric",
            args={"mode": "associations", "demographic_category": dem, "target_category": tgt},
        )
        for dem, tgt in cross_dem_target
    ] + [
        MetricSpec(
            class_name="helm.benchmark.metrics.bias_metrics.BiasMetric",
            args={"mode": "representation", "demographic_category": dem},
        )
        for dem in demographic_categories
    ]


def get_generative_harms_metric_specs(include_basic_metrics: bool = False) -> List[MetricSpec]:
    return (
        get_bias_metric_specs()
        + get_toxicity_metric_specs()
        + (get_basic_metric_specs([]) if include_basic_metrics else [])
    )


def get_summarization_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args=args)
    ] + get_basic_metric_specs([])


def get_summarization_critique_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.summarization_critique_metrics.SummarizationCritiqueMetric",
            args={"num_respondents": num_respondents},
        )
    ]


def get_srn_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["f1_set_match", "iou_set_match", "exact_set_match"])


def get_numeracy_metric_specs(run_solver: bool = False) -> List[MetricSpec]:
    metric_specs: List[MetricSpec] = get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "absolute_value_difference"]
    )

    # The solvers are slow to run so make them skippable
    if run_solver:
        metric_specs += [
            MetricSpec(class_name="helm.benchmark.metrics.numeracy_metrics.DistanceMetric", args={}),
        ]
    return metric_specs


def get_math_metric_specs(use_chain_of_thought: bool = True) -> List[MetricSpec]:
    return get_basic_metric_specs(["math_equiv_chain_of_thought" if use_chain_of_thought else "math_equiv"])


def get_copyright_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "longest_common_prefix_length"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "edit_distance"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "edit_similarity"},
        ),
    ] + get_basic_metric_specs([])


def get_disinformation_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationHumanEvalMetrics", args={**args}
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric", args={"name": "self_bleu"}
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric",
            args={"name": "monte_carlo_entropy"},
        ),
    ] + get_basic_metric_specs([])


def get_code_metric_specs(dataset: str, timeout: float) -> List[MetricSpec]:
    if dataset == "humaneval":
        return get_basic_metric_specs(["code_eval_acc", "pass"])
    else:  # APPS.
        args: Dict[str, Any] = {"names": ["test_avg", "strict_acc"], "timeout": timeout}
        return [MetricSpec(class_name="helm.benchmark.metrics.code_metrics.APPSMetric", args=args)]


def get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"])


def get_machine_translation_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.machine_translation_metrics.MachineTranslationMetric", args={})
    ] + get_basic_metric_specs([])


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


def get_verifiability_judgment_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match"])


def get_instruction_following_critique_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.instruction_following_critique_metrics.InstructionFollowingCritiqueMetric",  # noqa E501
            args={"num_respondents": num_respondents},
        )
    ]


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


############################################################
# Run specs


CANONICAL_RUN_SPEC_FUNCS: Dict[str, Callable[..., RunSpec]] = {}
"""Dict of run spec function names to run spec functions."""


F = TypeVar("F", bound=Callable[..., RunSpec])


def run_spec_function(name: str) -> Callable[[F], F]:
    """Register the run spec function under the given name."""

    def wrap(func: F) -> F:
        if name in CANONICAL_RUN_SPEC_FUNCS:
            raise ValueError(f"A run spec function with name {name} already exists")
        CANONICAL_RUN_SPEC_FUNCS[name] = func
        return func

    return wrap


@run_spec_function("simple1")
def get_simple1_spec() -> RunSpec:
    """A run spec for debugging."""
    return RunSpec(
        name="simple1",
        scenario_spec=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metric_specs=get_basic_metric_specs([]),
        groups=[],
    )


@run_spec_function("bbq")
def get_bbq_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bbq_scenario.BBQScenario", args={"subject": subject}
    )
    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers).",
        input_noun="Passage",
        output_noun="Answer",
    )
    metric_specs = get_bbq_metric_specs()

    return RunSpec(
        name=f"bbq:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["bbq"],
    )


@run_spec_function("msmarco")
def get_msmarco_spec(track: str, valid_topk: Optional[int] = None) -> RunSpec:
    valid_topk = None if valid_topk is None else int(valid_topk)
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.msmarco_scenario.MSMARCOScenario",
        args={"track": track, "valid_topk": valid_topk},
    )

    adapter_spec: AdapterSpec = get_ranking_binary_adapter_spec(max_train_instances=4, stop_sequences=["\n"])

    return RunSpec(
        name=f"msmarco:track={track},valid_topk={valid_topk}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_msmarco_metric_specs(track=track, rank=valid_topk),
        groups=[f"msmarco_{track}"],
    )


@run_spec_function("bold")
def get_bold_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bold_scenario.BOLDScenario", args={"subject": subject}
    )

    adapter_spec = get_completion_adapter_spec(
        temperature=0.9,  # Set to approximate nucleus sampling conditions.
        max_tokens=20,  # See Table 8 of RealToxicityPrompts: https://arxiv.org/pdf/2009.11462.pdf
    )

    return RunSpec(
        name=f"bold:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_generative_harms_metric_specs(include_basic_metrics=True),
        groups=["bold"],
    )


@run_spec_function("civil_comments")
def get_civil_comments_spec(demographic: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.civil_comments_scenario.CivilCommentsScenario",
        args={"demographic": demographic},
    )

    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Answer")

    return RunSpec(
        name=f"civil_comments:demographic={demographic}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_bias_metric_specs() + get_classification_metric_specs(),
        groups=["civil_comments"],
    )


@run_spec_function("custom_mcqa")
def get_custom_mcqa_spec(
    path: str,
    num_train_instances: int = 0,
    method: str = ADAPT_MULTIPLE_CHOICE_JOINT,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.custom_mcqa_scenario.CustomMCQAScenario",
        args={
            "path": path,
            "num_train_instances": num_train_instances,
        },
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers).",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=num_train_instances,
    )

    return RunSpec(
        name=f"custom_mcqa,path={path},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["custom"],
    )


@run_spec_function("mmlu")
def get_mmlu_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_scenario.MMLUScenario", args={"subject": subject}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"mmlu:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu"],
    )


@run_spec_function("interactive_qa_mmlu")
def get_interactive_qa_mmlu_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.interactive_qa_mmlu_scenario.InteractiveQAMMLUScenario",
        args={"subject": subject},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.",
        input_noun="Question",
        output_noun="Answer",
    )
    return RunSpec(
        name=f"interactive_qa_mmlu:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu"],
    )


@run_spec_function("wikifact")
def get_wikifact_spec(k: str, subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.wikifact_scenario.WIKIFactScenario",
        args={"subject": subject},
    )

    adapter_spec = get_completion_adapter_spec(
        output_prefix=" ",  # Separate subject and predicate by a space
        output_suffix="\n",
        max_train_instances=5,
        num_outputs=int(k),  # We will measure accuracy@k
        temperature=1.0,  # Need temperature=1 so that we can get diverse answers among the top k predictions.
        max_tokens=8,  # Number of tokens for the longest answer in the dataset
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"wikifact:k={k},subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=["wikifact"],
    )


@run_spec_function("commonsense")
def get_commonsense_spec(dataset: str, method: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.commonsense_scenario.CommonSenseScenario",
        args={"dataset": dataset},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers) about common sense.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"commonsense:dataset={dataset},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=[dataset],
    )


@run_spec_function("quac")
def get_quac_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.quac_scenario.QuACScenario", args={})

    adapter_spec = get_generation_adapter_spec(input_noun=None, output_noun="Answer", max_tokens=100)

    return RunSpec(
        name="quac",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["quac"],
    )


@run_spec_function("news_qa")
def get_news_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.newsqa_scenario.NewsQAScenario", args={})

    # max_tokens=50 because answers are at most 13 words
    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Answer", max_tokens=50)

    return RunSpec(
        name="news_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["news_qa"],
    )


@run_spec_function("truthful_qa")
def get_truthful_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.truthful_qa_scenario.TruthfulQAScenario",
        args={"task": task},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method, instructions="", input_noun="Question", output_noun="Answer"
    )

    return RunSpec(
        name=f"truthful_qa:task={task},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["truthful_qa"],
    )


@run_spec_function("twitter_aae")
def get_twitter_aae_spec(demographic: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.twitter_aae_scenario.TwitterAAEScenario",
        args={"demographic": demographic},
    )

    return RunSpec(
        name=f"twitter_aae:demographic={demographic}",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["twitter_aae", f"twitter_aae_{demographic}"],
    )


@run_spec_function("real_toxicity_prompts")
def get_real_toxicity_prompts_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.real_toxicity_prompts_scenario.RealToxicityPromptsScenario", args={}
    )

    # Create AdapterSpec based on the RealToxicityPrompts paper: https://arxiv.org/pdf/2009.11462.pdf
    adapter_spec = get_completion_adapter_spec(
        temperature=0.9,  # "We use a temperature of 1 during generation..."
        # Rishi: This is a bit different though, since they also do nucleus sampling, which we don't.
        max_tokens=20,  # "We generate up to 20 tokens per example..."
        # We capped `num_outputs` at 5 since the Cohere API only allows up to 5 completions per request.
        # Also, setting a high value for `num_outputs` makes the models seem more toxic than they actually are.
        # Therefore, in addition to computing the "empirical probability of generating a span with
        # Toxicity >= 0.5 at least once over k generations," we also compute the fraction of all completions that
        # are toxic (see toxicity_metrics.py).
        num_outputs=5,  # "...the expected maximum toxicity over k = 25 generations..."
    )

    return RunSpec(
        name="real_toxicity_prompts",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_generative_harms_metric_specs(include_basic_metrics=True),
        groups=["real_toxicity_prompts"],
    )


@run_spec_function("synthetic_reasoning_natural")
def get_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.synthetic_reasoning_natural_scenario.SRNScenario",
        args={"difficulty": difficulty},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Please solve the following problem.",
        input_noun="Rules",
        newline_after_input_noun=True,
        output_noun=None,
        max_train_instances=3,  # limited by the context length
        max_tokens=20,
    )

    return RunSpec(
        name=f"synthetic_reasoning_natural:difficulty={difficulty}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_srn_metric_specs() + get_generative_harms_metric_specs(),
        groups=["synthetic_reasoning", "synthetic_reasoning_natural"],
    )


@run_spec_function("gsm")
def get_gsm_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})

    # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
    adapter_spec = get_generation_adapter_spec(
        input_noun="Q",
        output_noun="A",
        max_train_instances=5,  # Due to limited context and long example length
        max_tokens=400,  # The paper uses 400 tokens as the max sample length
        stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
    )

    return RunSpec(
        name="gsm",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match_indicator"]) + get_generative_harms_metric_specs(),
        groups=["gsm"],
    )


@run_spec_function("raft")
def get_raft_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.raft_scenario.RAFTScenario", args={"subset": subset}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_raft_instructions(subset),
        input_noun=None,
        output_noun="Label",
        max_tokens=30,  # at most ~50 characters per label
    )

    return RunSpec(
        name=f"raft:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_bias_metric_specs() + get_classification_metric_specs(),
        groups=["raft"],
    )


@run_spec_function("numeracy")
def get_numeracy_spec(
    relation_type: str = "linear", mode: str = "function", seed: str = "0", run_solver: str = "False"
) -> RunSpec:
    from .scenarios.numeracy_scenario import get_numeracy_adapter_spec, RELTYPE_INFO

    run_solver: bool = True if run_solver == "True" else False  # type: ignore
    random_seed = int(seed)
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.numeracy_scenario.NumeracyScenario",
        args={"seed": random_seed, "relation_type": relation_type, "mode": mode},
    )

    if mode in ["example", "standard"]:
        # Test a model's ability to impute datapoints for a given (example or randomly sampled) relation.
        adapter_args: Dict[str, Any] = {
            "max_train_instances": 100,
            "max_eval_instances": 100,
            "dim": RELTYPE_INFO[relation_type].num_variables + 1,
        }
    elif mode == "function":
        # Test a model's ability to impute datapoints for randomly sampled relations
        # (resampled for each evaluation point).
        adapter_args = {
            "instructions": "",
            "max_train_instances": 0,  # Turn off general version of `function` mode because it doesn't cleanly
            # capture a higher-order version of this task / is a little convoluted
            # for models, currently.
            # (In the general version, the model sees other relations of the same class,
            # and needs to impute a datapoint for the last one. Presumably, inferring
            # the class - eg. the degree of the relation - would help.)
            "max_eval_instances": 1000,
            "dim": RELTYPE_INFO[relation_type].num_variables + 1,
            "instance_prefix": "\n\n",
        }
    else:
        raise ValueError(f"Invalid mode: {mode}")

    adapter_spec = get_numeracy_adapter_spec(**adapter_args)  # Construct the AdapterSpec using a helper function.
    # `get_numeracy_adapter_spec` is defined in numeracy_scenario.py
    # because it is used within the scenario to construct the instances themselves.

    return RunSpec(
        name=f"numeracy:relation_type={relation_type},mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_numeracy_metric_specs(run_solver),  # type: ignore
        groups=["numeracy"],
    )


@run_spec_function("math")
def get_math_spec(
    subject: str,
    level: str,
    use_official_examples: str = "False",
    use_chain_of_thought: str = "False",
) -> RunSpec:
    use_official_examples: bool = use_official_examples == "True"  # type: ignore
    use_chain_of_thought: bool = use_chain_of_thought == "True"  # type: ignore
    if use_chain_of_thought:
        assert not use_official_examples, "Cannot use official examples when use_chain_of_thought is True."
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.math_scenario.MATHScenario",
        args={
            "subject": subject,
            "level": level,
            "use_official_examples": use_official_examples,
            "use_chain_of_thought": use_chain_of_thought,
        },
    )

    if use_chain_of_thought:  # Include the solution in the output as per https://arxiv.org/abs/2201.11903
        output_prefix = "Answer: "  # Don't include LaTeX '$' delimiters
        output_suffix = "\n"
        instance_prefix = "###\n"  # Don't include LaTeX '$' delimiters
        max_tokens = 400  # Increase the number of tokens to generate
        stop_sequences = ["###"]  # Break at the next instance; extraneous output will be stripped out
        groups = ["math_chain_of_thought"]
    else:
        output_prefix = "Answer: $"
        output_suffix = "$\n"
        instance_prefix = "###\n"
        max_tokens = 20
        stop_sequences = ["$"]  # Break at the nearest LaTeX closing delimiter
        groups = ["math_regular"]

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Given a mathematics problem, determine the answer. Simplify your answer as much as possible.\n",
        max_train_instances=8,
        num_outputs=1,
        temperature=0.0,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        input_prefix="Problem: ",
        input_suffix="\n",
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        instance_prefix=instance_prefix,
    )

    return RunSpec(
        name=f"math:subject={subject},level={level},"
        f"use_official_examples={use_official_examples},use_chain_of_thought={use_chain_of_thought}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_math_metric_specs(use_chain_of_thought) + get_generative_harms_metric_specs(),  # type: ignore
        groups=groups,
    )


@run_spec_function("boolq")
def get_boolq_spec(only_contrast=False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.boolq_scenario.BoolQScenario", args={"only_contrast": only_contrast}
    )

    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Answer")

    return RunSpec(
        name="boolq" + (":only_contrast=True" if only_contrast else ""),
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_bias_metric_specs(),
        groups=["boolq"],
    )


@run_spec_function("lsat_qa")
def get_lsat_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lsat_qa_scenario.LSATScenario", args={"task": task}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers).",
        input_noun="Passage",
        output_noun="Answer",
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"lsat_qa:task={task},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lsat_qa"],
    )


@run_spec_function("imdb")
def get_imdb_spec(only_contrast=False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.imdb_scenario.IMDBScenario", args={"only_contrast": only_contrast}
    )

    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Sentiment")

    return RunSpec(
        name="imdb" + (":only_contrast=True" if only_contrast else ""),
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["imdb"],
    )


@run_spec_function("babi_qa")
def get_babi_qa_spec(task: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.babi_qa_scenario.BabiQAScenario", args={"task": task}
    )

    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Answer")

    return RunSpec(
        name=f"babi_qa:task={task}",
        scenario_spec=scenario_spec,
        # Answers are 1-2 words (1 for all tasks except task 19)
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["babi_qa"],
    )


@run_spec_function("copyright")
def get_copyright_spec(
    datatag="pilot",
    temperature=0.2,
    max_tokens=1024,
    num_outputs=1,
    normalize_by_prefix_length=True,
    normalize_newline_space_tab=False,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.copyright_scenario.CopyrightScenario", args=dict(datatag=datatag)
    )

    adapter_spec = get_completion_adapter_spec(temperature=temperature, max_tokens=max_tokens, num_outputs=num_outputs)

    return RunSpec(
        name=f"copyright:datatag={datatag}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_copyright_metric_specs(
            {
                "normalize_by_prefix_length": normalize_by_prefix_length,
                "normalize_newline_space_tab": normalize_newline_space_tab,
            }
        )
        + get_generative_harms_metric_specs(),
        groups=["copyright_code" if datatag in datatag2hash_code else "copyright_text"],
    )


@run_spec_function("disinformation")
def get_disinformation_spec(capability: str = "reiteration", topic: Optional[str] = "covid") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.disinformation_scenario.DisinformationScenario",
        args={"capability": capability, "topic": topic},
    )

    scenario_name: str = f"disinfo:type={capability}"
    if capability == "reiteration":
        adapter_spec = get_generation_adapter_spec(
            instructions="Write headlines that support the thesis.",
            input_noun="Thesis",
            output_noun="Headline",
            # Justification: Inspection. max_train_instances = 0 or 1 led to worse generations. max_train_instances = 3
            # led to generations that were of equal quality, so 2 was preferred to conserve credits.
            max_train_instances=2,
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in the
            # Pull_Climate_Skepticism.ipynb notebook located at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Amplification/
            temperature=0.7,
            num_outputs=5,
            max_tokens=100,
        )
        metric_specs = get_generative_harms_metric_specs() + get_disinformation_metric_specs(
            args={"name": "reiteration"}
        )
        scenario_name += f",topic={topic}"
    elif capability == "wedging":
        adapter_spec = get_completion_adapter_spec(
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in all notebooks at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Wedging/
            temperature=0.7,
            num_outputs=5,
            # Justification: Inspection. Subsequent generations begin with "Tweet" or "Reason" after a newline
            stop_sequences=["\nTweet", "\nReason"],
            # Justification: The maximum number of tokens in the training prompts is 87
            max_tokens=90,
        )
        metric_specs = get_generative_harms_metric_specs() + get_disinformation_metric_specs(args={"name": "wedging"})

    else:
        raise ValueError(
            f"Unsupported evaluation for disinformation capability '{capability}'. "
            f"Please choose one of 'reiteration' or 'wedging'."
        )

    # Self-BLEU isn't defined for a single sequence.
    if adapter_spec.num_outputs <= 1 and "self_bleu" in {metric_spec.args.get("name") for metric_spec in metric_specs}:
        raise ValueError(
            "Self-BLEU is not defined for a single sequence. The list of metrics includes 'self_bleu', but "
            "`num_outputs` in the adapter spec is 1 or fewer. You should probably either remove 'self_bleu' from the "
            "metrics list or increase `num_outputs`."
        )

    return RunSpec(
        name=scenario_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["disinformation", f"disinformation_{capability}"],
    )


@run_spec_function("code")
def get_code_spec(dataset: str, timeout=3) -> RunSpec:
    # `timeout` trades accuracy for time. Used exclusively for APPS. Default from original APPS codebase.
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.code_scenario.CodeScenario", args={"dataset": dataset}
    )

    if dataset == "humaneval":
        adapter_spec = get_completion_adapter_spec(
            temperature=0.2,
            # Taken from the original OpenAI paper to prevent the further generation of irrelevant classes/functions
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint"],
            max_tokens=600,
        )
    else:  # apps.
        # Different in `stop_sequences`.
        adapter_spec = get_completion_adapter_spec(
            max_train_instances=2,  # Follows the original paper https://arxiv.org/pdf/2105.09938.pdf Appendix D.
            temperature=0.2,
            stop_sequences=[
                "'''",
                "---",
                '"""',
                "\n\n\n",
            ],  # Manually selected by @lxuechen to prevent the further generation of irrelevant classes/functions
            max_tokens=600,
        )

    return RunSpec(
        name=f"code:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_metric_specs(dataset, timeout) + get_generative_harms_metric_specs(),
        groups=[f"code_{dataset}"],
    )


@run_spec_function("natural_qa")
def get_natural_qa_spec(mode: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={"mode": mode}
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Question" if mode == "closedbook" else None,
        output_noun="Answer",
        max_tokens=300,  # answers are at most 65 words
    )

    return RunSpec(
        name=f"natural_qa:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=[f"natural_qa_{mode}"],
    )


@run_spec_function("the_pile")
def get_the_pile_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.the_pile_scenario.ThePileScenario", args={"subset": subset}
    )

    return RunSpec(
        name=f"the_pile:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["the_pile"],
    )


@run_spec_function("ice")
def get_ice_spec(**kwargs) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.ice_scenario.ICEScenario", args=kwargs)

    return RunSpec(
        name="ice" + (":" if len(kwargs) > 0 else "") + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())),
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["ice"],
    )


@run_spec_function("narrative_qa")
def get_narrativeqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.narrativeqa_scenario.NarrativeQAScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=100,  # max 30 words
    )

    return RunSpec(
        name="narrative_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["narrative_qa"],
    )


@run_spec_function("synthetic_efficiency")
def get_synthetic_efficiency_spec(
    num_prompt_tokens: Optional[int] = None,
    num_output_tokens: Optional[int] = None,
    tokenizer: Optional[str] = None,
    random: Optional[str] = None,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario",
        args={"num_prompt_tokens": num_prompt_tokens, "num_instances": 10, "tokenizer": tokenizer},
    )

    if num_output_tokens is not None:
        adapter_spec = get_completion_adapter_spec(max_tokens=num_output_tokens, random=random)
    else:
        adapter_spec = get_completion_adapter_spec(random=random)

    return RunSpec(
        name=f"synthetic_efficiency:random={random}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match"]) + get_generative_harms_metric_specs(),
        groups=["synthetic_efficiency"],
    )


@run_spec_function("synthetic_reasoning")
def get_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.synthetic_reasoning_scenario.SyntheticReasoningScenario",
        args={"mode": mode},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Please solve the following problem.",
        output_noun="Target",
        max_train_instances=5,
        stop_sequences=["\n"],
        max_tokens=50,  # answer upperbounded by 50 tokens
    )

    return RunSpec(
        name=f"synthetic_reasoning:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=["synthetic_reasoning", f"synthetic_reasoning_{mode}"],
    )


@run_spec_function("wikitext_103")
def get_wikitext_103_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.wikitext_103_scenario.Wikitext103Scenario", args={}
    )

    return RunSpec(
        name="wikitext_103",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["wikitext_103"],
    )


@run_spec_function("blimp")
def get_blimp_spec(phenomenon: str, method: str = ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.blimp_scenario.BLiMPScenario", args={"phenomenon": phenomenon}
    )
    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="Please select the grammatical sentence.",
        input_noun=None,
        output_noun="Answer",
        empty_input=True,
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"blimp:phenomenon={phenomenon},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["blimp"],
    )


@run_spec_function("summarization_xsum")
def get_xsum_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "xsum", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_xsum", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["summarization_xsum"],
    )


@run_spec_function("summarization_xsum_sampled")
def get_xsum_sampled_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={
            "dataset_name": "xsum-sampled",
            "sampling_min_length": 50,
            "sampling_max_length": 150,
            "doc_max_length": 512,
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_xsum_sampled", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["summarization_xsum"],
    )


@run_spec_function("summarization_cnndm")
def get_cnndm_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "cnn-dm", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=3,
        max_tokens=128,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # From Wu et al. 2021 (https://arxiv.org/pdf/2109.10862.pdf)
    )

    return RunSpec(
        name=f"summarization_cnndm:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_cnndm", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["summarization_cnndm"],
    )


@run_spec_function("empatheticdialogues")
def get_empatheticdialogues_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.dialogue_scenarios.EmpatheticDialoguesScenario", args={}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="BEGIN DIALOGUE\n",
        max_train_instances=5,
        num_outputs=1,
        max_tokens=50,  # TODO: Justify
        temperature=0.9,  # TODO: Justify
        # TODO: Add stop sequences
    )

    return RunSpec(
        name="empatheticdialogues",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=[],
    )


@run_spec_function("dyck_language")
def get_dyck_language_spec(num_parenthesis_pairs: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.dyck_language_scenario.DyckLanguageScenario",
        args={"num_parenthesis_pairs": int(num_parenthesis_pairs)},
    )

    adapter_spec = get_completion_adapter_spec(
        instructions="Please complete the rest of the following Dyck sequences, "
        "making sure that the parentheses are closed properly.",
        input_prefix="Input: ",
        max_tokens=5,
        max_train_instances=3,  # Determined by looking at average length of examples to see what fits
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"dyck_language_np={int(num_parenthesis_pairs)}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match_indicator"]) + get_generative_harms_metric_specs(),
        groups=["dyck_language"],
    )


@run_spec_function("legal_support")
def get_legal_support_spec(method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_support_scenario.LegalSupportScenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="Which statement best supports the passage?",
        input_noun="Passage",
        output_noun="Answer",
        max_train_instances=3,  # We use 3 because these samples tend to be a bit longer
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"legal_support,method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["legal_support"],
    )


@run_spec_function("entity_matching")
def get_entity_matching_spec(dataset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.entity_matching_scenario.EntityMatchingScenario", args={"dataset": dataset}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Are Product A and Product B the same? Yes or No?",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"entity_matching:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=["entity_matching"],
    )


@run_spec_function("entity_data_imputation")
def get_entity_data_imputation_spec(dataset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.entity_data_imputation_scenario.EntityDataImputationScenario",
        args={"dataset": dataset},
    )

    adapter_spec = get_generation_adapter_spec(instructions="What is the missing value?", output_noun="Answer")

    return RunSpec(
        name=f"entity_data_imputation:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_generative_harms_metric_specs(),
        groups=["entity_data_imputation"],
    )


@htrack("Extracting adaptation parameters from the BIG-bench task definition and building the RunSpec")
@run_spec_function("big_bench")
def get_big_bench_spec(task: str, subtask: str) -> RunSpec:
    def get_adaptation_method(big_bench_metrics: List[str]) -> str:
        """
        From BIG-bench, "there are three types of BIG-bench JSON tasks - generative and scoring
        (e.g. simple_arithmetic_json), and multiple-choice (e.g. simple_arithmetic_json_multiple_choice)."

        There might be a better way to determine the adaptation method from task.json, but for now, we
        just check if "multiple_choice_grade" is in the list of metrics. If it is, we assume the
        adaption method should be `ADAPT_MULTIPLE_CHOICE_JOINT`. Otherwise, the adaptation method is
        `ADAPT_GENERATION`.
        """
        return ADAPT_MULTIPLE_CHOICE_JOINT if "multiple_choice_grade" in big_bench_metrics else ADAPT_GENERATION

    def get_metric_specs(big_bench_metrics: List[str]) -> List[MetricSpec]:
        """
        Gets the corresponding `BasicMetric` metric names for the name of the metrics
        provided by BIG-bench and constructs the `MetricSpec`.

        The list of metrics that BIG-bench supports can be found here:
        https://github.com/google/BIG-bench/blob/main/docs/doc.md#available-metrics.
        """
        metric_names: Set[str] = set()

        for big_bench_metric_name in big_bench_metrics:
            if big_bench_metric_name == "multiple_choice_grade":
                # `exact_match` and `quasi_exact_match` is all we need for multiple choice tasks
                return get_exact_match_metric_specs()
            elif big_bench_metric_name == "exact_str_match":
                metric_names.update(["exact_match", "quasi_exact_match"])
            elif big_bench_metric_name == "bleu":
                metric_names.update(["bleu_1", "bleu_4"])
            elif big_bench_metric_name == "rouge":
                metric_names.update(["rouge_1", "rouge_2", "rouge_l"])
            else:
                hlog(f"Unhandled BIG-bench metric: {big_bench_metric_name}")
                continue

        return get_basic_metric_specs(list(metric_names))

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.big_bench_scenario.BIGBenchScenario",
        args={"task": task, "subtask": subtask},
    )

    # Get BIG-bench task definition.
    # TODO: get `output_path` here without hardcoding
    output_path: str = "benchmark_output/scenarios/big_bench"
    big_bench_task: Dict = BIGBenchScenario.download_and_get_task(output_path, task, subtask)

    # The JSON schema for BIG-bench can be found here:
    # https://github.com/google/BIG-bench/blob/main/docs/doc.md#json-schema.
    # "metrics" is a required field. The default values were populated using the link above.
    adapter_spec = AdapterSpec(
        method=get_adaptation_method(big_bench_task["metrics"]),
        model="openai/text-curie-001",  # Can override with the `ModelRunExpander`.
        max_train_instances=5,  # Can override with the `MaxTrainInstancesRunExpander`.
        num_outputs=1,  # Can override with the `NumOutputsRunExpander`.
        # From "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models",
        # for the BIG-G models tested on BIG-bench, "we use an input context length of 1,024 tokens
        # and an output length of 64 tokens. We evaluate on up to 1,000 examples per task".
        max_tokens=64,
        # "all model outputs were sampled greedily (with zero temperature), unless otherwise noted."
        temperature=0,
        instructions=big_bench_task.get("task_prefix", ""),
        # BIG-bench's default value for "example_input_prefix" and "example_output_prefix" was "\nQ: " and "\nA: ".
        # Instead, use our defaults for multiple choice tasks: "Question: " and "\nAnswer: ".
        input_prefix=big_bench_task.get("example_input_prefix", "Question: "),
        output_prefix=big_bench_task.get("example_output_prefix", "Answer: "),
        # Use our default for multiple choice: A., B., C., D.,...
        # reference_prefix=big_bench_task.get("choice_prefix", "\n choice: "),
        # The default value for "stop_string" in BIG-bench is None.
        stop_sequences=[str(big_bench_task.get("stop_string"))] if big_bench_task.get("stop_string", None) else [],
    )

    run_spec_name: str = f"big_bench:task={task}"
    if subtask:
        run_spec_name += f",subtask={subtask}"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        # TODO add generative harms when applicable
        metric_specs=get_metric_specs(big_bench_task["metrics"]),
        groups=["BIG-bench"],
    )


@run_spec_function("covid_dialog")
def get_covid_dialog_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.covid_dialog_scenario.COVIDDialogScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Generate a response given a patient's questions and concerns.",
        input_noun="Patient",
        output_noun="Doctor",
        max_tokens=128,
    )

    return RunSpec(
        name="covid_dialog",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["COVIDDialog"],
    )


@run_spec_function("me_q_sum")
def get_me_q_sum_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.me_q_sum_scenario.MeQSumScenario", args={})

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=128,
        temperature=0.3,
    )

    return RunSpec(
        name="me_q_sum",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MeQSum"],
    )


@run_spec_function("med_dialog")
def get_med_dialog_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_dialog_scenario.MedDialogScenario", args={"subset": subset}
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=128,
        temperature=0.3,
    )

    return RunSpec(
        name=f"med_dialog,subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MedDialog"],
    )


@run_spec_function("med_mcqa")
def get_med_mcqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.med_mcqa_scenario.MedMCQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Give a letter answer among A, B, C or D.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="med_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["MedMCQA"],
    )


@run_spec_function("med_paragraph_simplification")
def get_med_paragraph_simplification_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_paragraph_simplification_scenario.MedParagraphSimplificationScenario",
        args={},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=10,
        max_tokens=512,
        temperature=0.3,
    )

    return RunSpec(
        name="med_paragraph_simplification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MedParagraphSimplification"],
    )


@run_spec_function("med_qa")
def get_med_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.med_qa_scenario.MedQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Give a letter answer among A, B, C or D.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="med_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["MedQA"],
    )


@run_spec_function("pubmed_qa")
def get_pubmed_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.pubmed_qa_scenario.PubMedQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer A for yes, B for no or C for maybe.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="pubmed_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["pubmed_qa"],
    )


@run_spec_function("lextreme")
def get_lextreme_spec(subset: str) -> RunSpec:
    task_type = get_lextreme_task_type(subset)

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lextreme_scenario.LEXTREMEScenario",
        args={"subset": subset},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_lextreme_instructions(subset),
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=get_lextreme_max_tokens(subset),
        max_train_instances=get_lextreme_max_train_instances(subset),  # in some subsets the input is very long
        multi_label=(task_type == TaskType.MLTC),
    )

    metric_specs = get_basic_metric_specs([])
    if task_type == TaskType.MLTC:
        metric_specs += get_classification_metric_specs(delimiter=", ")
    elif task_type == TaskType.SLTC:
        metric_specs += get_classification_metric_specs()

    return RunSpec(
        name=f"lextreme:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lextreme"],
    )


@run_spec_function("lex_glue")
def get_lex_glue_spec(subset: str) -> RunSpec:
    task_type = get_lex_glue_task_type(subset)

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lex_glue_scenario.LexGLUEScenario",
        args={"subset": subset},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_lex_glue_instructions(subset),
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=get_lex_glue_max_tokens(subset),
        max_train_instances=get_lex_glue_max_train_instances(subset),  # in some subsets the input is very long
        multi_label=(task_type == TaskType.MLTC),
    )

    metric_specs = get_basic_metric_specs([])
    if task_type == TaskType.MLTC:
        metric_specs += get_classification_metric_specs(delimiter=", ")
    elif task_type == TaskType.SLTC:
        metric_specs += get_classification_metric_specs()

    return RunSpec(
        name=f"lex_glue:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lex_glue"],
    )


@run_spec_function("billsum_legal_summarization")
def get_billsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "BillSum",
            "sampling_min_length": 200,
            "sampling_max_length": 800,  # 2000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 2048,  # 4096 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=None,
        max_tokens=1024,  # From Kornilova & Eidelmann, 2020 (https://arxiv.org/pdf/1910.00523.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "billsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )


@run_spec_function("multilexsum_legal_summarization")
def get_multilexsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "MultiLexSum",
            "sampling_min_length": 100,
            "sampling_max_length": 400,  # 1000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 1024,  # 2048 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=2,
        max_tokens=256,  # From Shen et al., 2022 (https://arxiv.org/pdf/2206.10883.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "multilexsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )


@run_spec_function("eurlexsum_legal_summarization")
def get_eurlexsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "EurLexSum",
            "sampling_min_length": 400,
            "sampling_max_length": 1600,  # 4000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 2048,  # 8192 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=None,
        max_tokens=2048,  # From Aumiller et al., 2022 (https://arxiv.org/pdf/2210.13448.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "eurlexsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )


@run_spec_function("wmt_14")
def get_wmt_14_spec(language_pair: str, max_train_instances: int = 1) -> RunSpec:
    FULL_LANGUAGE_NAMES = {
        "cs": "Czech",
        "de": "German",
        "fr": "French",
        "hi": "Hindi",
        "ru": "Russian",
        "en": "English",
    }
    source_language, target_language = language_pair.split("-")

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.wmt_14_scenario.WMT14Scenario",
        args={"source_language": source_language, "target_language": target_language},
    )

    adapter_spec = get_machine_translation_adapter_spec(
        source_language=FULL_LANGUAGE_NAMES[source_language],
        target_language=FULL_LANGUAGE_NAMES[target_language],
        max_train_instances=max_train_instances,
    )

    return RunSpec(
        name=f"wmt_14:language_pair={language_pair}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_machine_translation_metric_specs(),
        groups=["wmt_14"],
    )


@run_spec_function("self_instruct")
def get_self_instruct_spec(num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.self_instruct_scenario.SelfInstructScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name="self_instruct",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["self_instruct"],
    )


@run_spec_function("vicuna")
def get_vicuna_spec(num_respondents: int, category: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vicuna_scenario.VicunaScenario",
        args={"category": category},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"vicuna:category={category}",  # TODO: add args
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["vicuna"],
    )


@run_spec_function("grammar")
def get_grammar_spec(num_respondents: int, path: str, tags: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.grammar_scenario.GrammarScenario",
        args={"path": path, "tags": tags},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"grammar:path={path},tags={tags}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["grammar"],
    )


@run_spec_function("verifiability_judgment")
def get_verifiability_judgment_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.verifiability_judgment_scenario.VerifiabilityJudgementScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=(
            'Given the statement and its source, judge whether the source "fully supports", '
            '"partially supports" or "does not support" the statement.'
        ),
        input_noun="Statement",
        # Add another new line before the output noun, since the source might have
        # newlines embedded in it.
        output_noun="\nJudgment",
        max_tokens=10,
    )

    return RunSpec(
        name="verifiability_judgment",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_verifiability_judgment_metric_specs(),
        groups=["verifiability_judgment"],
    )


@run_spec_function("opinions_qa")
def get_opinions_qa_spec(
    survey_type: str,
    num_logprobs: str,
    context: str = "None",
    num_train_trials: str = "1",
    method: str = ADAPT_MULTIPLE_CHOICE_JOINT,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.opinions_qa_scenario.OpinionsQAScenario",
        args={"survey_type": survey_type, "context": context},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=1 if "steer" in context else 0,
        max_tokens=1,
        num_outputs=int(num_logprobs),
        num_train_trials=1 if context != "steer-qa" else int(num_train_trials),
        sample_train=False,
    )

    return RunSpec(
        name=f"opinions_qa:survey={survey_type},num_logprobs={num_logprobs}"
        + f",context={context},num_train_trials={num_train_trials}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=[],
        groups=["opinions_qa"],
    )


@run_spec_function("open_assistant")
def get_open_assistant_spec(num_respondents: int, language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.open_assistant_scenario.OpenAssistantScenario",
        args={"language": language},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"open_assistant:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["open_assistant"],
    )


@run_spec_function("koala")
def get_koala_spec(num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.koala_scenario.KoalaScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name="koala",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["koala"],
    )


@run_spec_function("anthropic_hh_rlhf")
def get_anthropic_hh_rlhf_spec(num_respondents: int, subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.anthropic_hh_rlhf_scenario.AnthropicHHRLHFScenario",
        args={"subset": subset},
    )

    adapter_spec = get_instruct_adapter_spec()

    return RunSpec(
        name=f"anthropic_hh_rlhf:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["anthropic_hh_rlhf"],
    )


@run_spec_function("cleva")
def get_cleva_spec(task: str, version: str, subtask: Optional[str] = None, prompt_id: int = 0) -> RunSpec:
    from .scenarios.cleva_scenario import CLEVAScenario  # noqa

    CLEVAScenario.download_dataset(task, version)

    _, prompt_setting = CLEVAScenario.get_prompt_setting(task, subtask, version, prompt_id)
    inference_parameters = CLEVAScenario.load_inference_parameters(task, subtask, version, prompt_id)

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
        metric_specs = get_basic_metric_specs(["code_eval_acc", "pass"]) + get_cleva_generative_harms_metric_specs()
    elif task in ["language_modeling"]:
        adapter_spec = get_language_modeling_adapter_spec()
        metric_specs = get_basic_metric_specs([])
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


############################################################


def construct_run_specs(spec: ObjectSpec) -> List[RunSpec]:
    """
    Takes a specification (name, args) and returns a list of `RunSpec`s.
    """
    # Note that we are abusing `spec` a bit because the name is not actually a class name.
    name = spec.class_name
    args = spec.args

    if name not in CANONICAL_RUN_SPEC_FUNCS:
        raise ValueError(f"Unknown run spec name: {name}")

    # Peel off the run expanders (e.g., model)
    expanders = [RUN_EXPANDERS[key](value) for key, value in args.items() if key in RUN_EXPANDERS]  # type: ignore
    args = dict((key, value) for key, value in args.items() if key not in RUN_EXPANDERS)

    # Get the canonical run specs
    run_specs = [CANONICAL_RUN_SPEC_FUNCS[name](**args)]

    # Apply expanders
    for expander in expanders:
        run_specs = [
            child_run_spec for parent_run_spec in run_specs for child_run_spec in expander.expand(parent_run_spec)
        ]

    def alter_run_spec(run_spec: RunSpec) -> RunSpec:
        try:
            model = get_model(run_spec.adapter_spec.model)
        except ValueError:
            # Models registered from configs cannot have expanders applied to them,
            # because the models will not have been registered yet at this point.
            # TODO: Figure out a cleaner way to deal with this.
            return run_spec
        # For models that strip newlines, when we're generating, we need to set
        # the delimiter to be '###' so we stop properly.
        if NO_NEWLINES_TAG in model.tags and run_spec.adapter_spec.method in (
            ADAPT_GENERATION,
            ADAPT_MULTIPLE_CHOICE_JOINT,
        ):
            stop_expander = StopRunExpander(value="hash")
            run_spec = singleton(stop_expander.expand(run_spec))

        if NLG_PREFIX_TAG in model.tags:
            global_prefix_expander = GlobalPrefixRunExpander(value="nlg")
            run_spec = singleton(global_prefix_expander.expand(run_spec))

        # When running ChatGPT on non-language modelling tasks, increase max_tokens by 1
        # to add room for the special message role token.
        if OPENAI_CHATGPT_MODEL_TAG in model.tags and run_spec.adapter_spec.max_tokens:
            increase_max_tokens_expander = IncreaseMaxTokensRunExpander(value=1)
            run_spec = singleton(increase_max_tokens_expander.expand(run_spec))

        if CHATML_MODEL_TAG in model.tags:
            chatml_expander = ChatMLRunExpander()
            run_spec = singleton(chatml_expander.expand(run_spec))

        # Special handling for Anthropic Claude
        if ANTHROPIC_CLAUDE_1_MODEL_TAG in model.tags or ANTHROPIC_CLAUDE_2_MODEL_TAG in model.tags:
            try:
                import anthropic
                from helm.proxy.clients.anthropic_client import AnthropicClient
            except ModuleNotFoundError as e:
                handle_module_not_found_error(e, ["anthropic"])
            claude_run_expanders: List[RunExpander] = []
            claude_run_expanders.append(AddToStopRunExpander(anthropic.HUMAN_PROMPT))
            if ANTHROPIC_CLAUDE_1_MODEL_TAG in model.tags:
                claude_run_expanders.append(IncreaseMaxTokensRunExpander(value=AnthropicClient.ADDITIONAL_TOKENS))
            # Get scenario tags
            components = run_spec.scenario_spec.class_name.split(".")
            class_name = components[-1]
            module_name = ".".join(components[:-1])
            cls = getattr(importlib.import_module(module_name), class_name)
            scenario_tags: List[str] = cls.tags
            # If the scenario is instruction, do not use PROMPT_ANSWER_START
            if "instructions" in scenario_tags:
                claude_run_expanders.append(
                    FormatPromptRunExpander(prefix=anthropic.HUMAN_PROMPT, suffix=f"{anthropic.AI_PROMPT}")
                )
            else:
                claude_run_expanders.append(
                    FormatPromptRunExpander(
                        prefix=anthropic.HUMAN_PROMPT,
                        suffix=f"{anthropic.AI_PROMPT} {AnthropicClient.PROMPT_ANSWER_START}",
                    )
                )
            for claude_run_expander in claude_run_expanders:
                run_spec = singleton(claude_run_expander.expand(run_spec))

        # For multiple choice
        if BUGGY_TEMP_0_TAG in model.tags and run_spec.adapter_spec.temperature == 0:
            increase_temperature_expander = IncreaseTemperatureRunExpander(value=1e-4)
            run_spec = singleton(increase_temperature_expander.expand(run_spec))

        return run_spec

    run_specs = [alter_run_spec(run_spec) for run_spec in run_specs]

    return run_specs
