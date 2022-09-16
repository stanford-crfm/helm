import itertools
import os
from typing import Any, Callable, List, Dict, Optional, Set

from common.hierarchical_logger import hlog, htrack
from common.object_spec import ObjectSpec
from .adapter import (
    AdapterSpec,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_GENERATION,
)
from .metrics.metric import MetricSpec
from .run_expander import RUN_EXPANDERS
from .runner import RunSpec
from .scenarios.scenario import ScenarioSpec
from .scenarios.big_bench_scenario import BIGBenchScenario
from .scenarios.msmarco_scenario import MSMARCOScenario
from .scenarios.numeracy_scenario import get_numeracy_adapter_spec, RELTYPE_INFO
from .scenarios.raft_scenario import get_raft_instructions


############################################################
# Prototypical adapter specs


def get_multiple_choice_joint_adapter_spec(
    instructions: str, input_noun: str, output_noun: str, max_train_instances: int = 5, **kwargs
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=f"{instructions}\n",
        input_prefix=f"{input_noun}: ",
        input_suffix="\n",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=5,
        temperature=0.0,
        stop_sequences=["\n"],
        **kwargs,
    )


def get_multiple_choice_separate_adapter_spec(method: str) -> AdapterSpec:
    assert method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}

    return AdapterSpec(
        method=method,
        instructions="",
        input_prefix="",
        output_prefix=" ",  # Note the space
        # Separate is basically language modeling, so can't easily use in-context learning examples
        max_train_instances=0,
        num_outputs=1,
        max_tokens=0,
        temperature=0.0,
    )


def get_multiple_choice_adapter_spec(
    method: str, instructions: str, input_noun: str, output_noun: str, max_train_instances: int = 5, **kwargs
):
    """
    Toggle between different adapters.
    """
    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        return get_multiple_choice_joint_adapter_spec(
            instructions, input_noun, output_noun, max_train_instances, **kwargs
        )
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        return get_multiple_choice_separate_adapter_spec(method)
    else:
        raise ValueError(f"Invalid adaptation method: {method}")


def get_multiple_choice_joint_empty_input_adapter_spec(
    input_instructions: str, output_noun: str, max_train_instances: int = 5, **kwargs
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="",
        input_prefix=input_instructions,
        input_suffix="\n",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        max_train_instances=5,
        num_outputs=1,
        temperature=0.0,
        stop_sequences=["\n"],
        **kwargs,
    )


def get_multiple_completions_adapter_spec() -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
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


def get_completion_adapter_spec(**kwargs) -> AdapterSpec:
    # Given the input, complete it
    return AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        input_suffix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=1,
        **kwargs,
    )


def get_generation_adapter_spec(
    input_noun: Optional[str], output_noun: str, max_train_instances: int = 5, max_tokens: int = 5
) -> AdapterSpec:
    # Used for classification (e.g., sentiment classification)
    return AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix=f"{input_noun}: " if input_noun is not None else "",
        input_suffix="\n",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=["\n"],
    )


def get_language_modeling_adapter_spec() -> AdapterSpec:
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


############################################################
# Concrete adapter specs


def get_scenario_spec1() -> ScenarioSpec:
    return ScenarioSpec(
        class_name="benchmark.scenarios.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )


def get_scenario_spec_tiny():
    return ScenarioSpec(
        class_name="benchmark.scenarios.simple_scenarios.Simple1Scenario",
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
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": names})]


def get_exact_match_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match"])


def get_f1_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match", "f1_score"])


def get_bbq_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.bbq_metrics.BBQMetric", args={})] + get_exact_match_metric_specs()


def get_msmarco_metric_specs(task: str, track: str, qrels_path: str, topk: Optional[int] = None) -> List[MetricSpec]:
    measure_names = MSMARCOScenario.MEASURE_NAMES[(task, track)]
    mode = MSMARCOScenario.BINARY_LOGPROB_MODE
    correct_output, wrong_output = MSMARCOScenario.CORRECT_OUTPUT, MSMARCOScenario.WRONG_OUTPUT
    multi_value_qrels = set(MSMARCOScenario.GOLD_RELATIONS[(task, track)]) != {1}

    return [
        MetricSpec(
            class_name="benchmark.multiple_request_metrics.InformationRetrievalMetric",
            args={
                "measure_names": measure_names,
                "qrels_path": qrels_path,
                "mode": mode,
                "correct_output": correct_output,
                "wrong_output": wrong_output,
                "topk": topk,
                "multi_value_qrels": multi_value_qrels,
            },
        ),
        MetricSpec(
            class_name="benchmark.multiple_request_metrics.MultipleRequestMetrics", args={"use_basic_metrics": True}
        ),
        # The line below is commented out because efficiency metrics are taking a long time to compute
        # @TODO Uncomment the line below when we have the efficiency computations for all the models
        # MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": []}),
    ]


def get_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={}),
    ]


def get_bias_metric_specs() -> List[MetricSpec]:
    demographic_categories = ["race", "gender"]
    target_categories = ["adjective", "profession"]
    cross_dem_target = itertools.product(demographic_categories, target_categories)

    return [
        MetricSpec(
            class_name="benchmark.bias_metrics.BiasMetric",
            args={"mode": "associations", "demographic_category": dem, "target_category": tgt},
        )
        for dem, tgt in cross_dem_target
    ] + [
        MetricSpec(
            class_name="benchmark.bias_metrics.BiasMetric",
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
        MetricSpec(class_name="benchmark.summarization_metrics.SummarizationMetric", args=args)
    ] + get_basic_metric_specs([])


def get_srn_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["f1_set_match", "iou_set_match", "exact_set_match"])


def get_numeracy_metric_specs(run_solver: bool = False) -> List[MetricSpec]:
    metric_specs: List[MetricSpec] = get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "absolute_value_difference"]
    )

    # The solvers are slow to run so make them skippable
    if run_solver:
        metric_specs += [
            MetricSpec(class_name="benchmark.numeracy_metrics.DistanceMetric", args={}),
        ]
    return metric_specs


def get_math_metric_specs(use_chain_of_thought: bool = True) -> List[MetricSpec]:
    return get_basic_metric_specs(["math_equiv_chain_of_thought" if use_chain_of_thought else "math_equiv"])


def get_copyright_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="benchmark.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "longest_common_prefix_length"},
        ),
        MetricSpec(
            class_name="benchmark.copyright_metrics.BasicCopyrightMetric", args={**args, "name": "edit_distance"},
        ),
        MetricSpec(
            class_name="benchmark.copyright_metrics.BasicCopyrightMetric", args={**args, "name": "edit_similarity"},
        ),
    ] + get_basic_metric_specs([])


def get_disinformation_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(class_name="benchmark.disinformation_metrics.DisinformationHumanEvalMetrics", args={**args}),
        MetricSpec(class_name="benchmark.disinformation_metrics.DisinformationMetric", args={"name": "self_bleu"}),
        MetricSpec(
            class_name="benchmark.disinformation_metrics.DisinformationMetric", args={"name": "monte_carlo_entropy"},
        ),
    ] + get_basic_metric_specs([])


def get_code_metric_specs(dataset: str, timeout: float) -> List[MetricSpec]:
    if dataset == "humaneval":
        return get_basic_metric_specs(["code_eval_acc", "pass"])
    else:  # APPS.
        args: Dict[str, Any] = {"names": ["test_avg", "strict_acc"], "timeout": timeout}
        return [MetricSpec(class_name="benchmark.code_metrics.APPSMetric", args=args)]


############################################################


def get_simple1_spec() -> RunSpec:
    """An run spec for debugging."""
    return RunSpec(
        name="simple1",
        scenario_spec=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metric_specs=get_basic_metric_specs([]),
        groups=[],
    )


def get_bbq_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.bbq_scenario.BBQScenario", args={"subject": subject})

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        adapter_spec = get_multiple_choice_joint_adapter_spec(
            "The following are multiple choice questions (with answers).", "Passage", "Answer"
        )
        metric_specs = get_bbq_metric_specs()
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        adapter_spec = get_multiple_choice_separate_adapter_spec(method)
        # TODO: We do not compute BBQ metrics when non-standard method is used
        metric_specs = get_basic_metric_specs([])
    else:
        raise ValueError(f"Invalid adaptation method: {method}")

    return RunSpec(
        name=f"bbq:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["bbq"],
    )


def get_msmarco_spec(
    task,
    track,
    use_qrels_passages="False",
    use_topk_passages="False",
    valid_topk=None,
    num_valid_queries=None,
    num_train_queries="1000",
) -> RunSpec:

    # Get ScenarioSpec
    use_qrels_passages = use_qrels_passages.lower() == "true"
    use_topk_passages = use_topk_passages.lower() == "true"
    valid_topk = int(valid_topk) if valid_topk else valid_topk
    num_valid_queries = int(num_valid_queries) if num_valid_queries else num_valid_queries
    num_train_queries = int(num_train_queries)
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.msmarco_scenario.MSMARCOScenario",
        args={
            "task": task,
            "track": track,
            "use_qrels_passages": use_qrels_passages,
            "use_topk_passages": use_topk_passages,
            "valid_topk": valid_topk,
            "num_valid_queries": num_valid_queries,
            "num_train_queries": num_train_queries,
        },
    )

    # Get AdapterSpec
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="Passage: ",
        output_prefix="Answer: ",
        max_train_instances=4,  # Needs to be even to ensure equal number of correct and wrong examples
        num_outputs=1,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    # Create metrics
    qrels_path: str = os.path.join("benchmark_output", "scenarios", "msmarco", "data", f"{task}_{track}_qrels.tsv")

    # Return RunSpec
    return RunSpec(
        name=f"msmarco:task={task},track={track},use_qrels_passages={use_qrels_passages},"
        f"use_topk_passages={use_topk_passages},valid_topk={valid_topk},num_valid_queries={num_valid_queries},"
        f"num_train_queries={num_train_queries}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_msmarco_metric_specs(task, track, qrels_path, topk=valid_topk)
        + get_generative_harms_metric_specs(),
        groups=[f"msmarco_{track}"],
    )


def get_bold_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.bold_scenario.BOLDScenario", args={"subject": subject})

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


def get_civil_comments_spec(demographic: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.civil_comments_scenario.CivilCommentsScenario",
        args={"demographic": demographic},
    )

    return RunSpec(
        name=f"civil_comments:demographic={demographic}",
        scenario_spec=scenario_spec,
        adapter_spec=get_generation_adapter_spec("Passage", "Answer"),
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["civil_comments"],
    )


def get_mmlu_spec(subject: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.mmlu_scenario.MMLUScenario", args={"subject": subject})

    def format(subject: str):
        return subject.replace("_", " ")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=f"The following are multiple choice questions (with answers) about {format(subject)}.",
        input_noun="Question",
        output_noun="Answer",
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"mmlu:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["mmlu"],
    )


def get_wikifact_spec(k: str, subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.wikifact_scenario.WIKIFactScenario", args={"subject": subject},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        input_suffix="",
        output_prefix=" ",  # Separate subject and predicate by a space
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
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["wikifact"],
    )


def get_commonsense_spec(dataset: str, method: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.commonsense_scenario.CommonSenseScenario", args={"dataset": dataset},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers) about common sense.",
        input_noun="Question",
        output_noun="Answer",
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"commonsense:dataset={dataset},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[dataset],
    )


def get_quac_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.quac_scenario.QuACScenario", args={})

    adapter_spec = get_generation_adapter_spec(input_noun=None, output_noun="Answer", max_tokens=100)
    return RunSpec(
        name="quac",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["quac"],
    )


def get_news_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.newsqa_scenario.NewsQAScenario", args={})

    # Answers are at most 13 words
    adapter_spec = get_generation_adapter_spec(input_noun="Passage", output_noun="Answer", max_tokens=50)

    return RunSpec(
        name="news_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["news_qa"],
    )


def get_truthful_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.truthful_qa_scenario.TruthfulQAScenario", args={"task": task},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method, instructions="", input_noun="Question", output_noun="Answer"
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"truthful_qa:task={task}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["truthful_qa"],
    )


def get_twitter_aae_spec(demographic: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.twitter_aae_scenario.TwitterAAEScenario", args={"demographic": demographic},
    )

    return RunSpec(
        name=f"twitter_aae:demographic={demographic}",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["twitter_aae", f"twitter_aae_{demographic}"],
    )


def get_real_toxicity_prompts_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.real_toxicity_prompts_scenario.RealToxicityPromptsScenario", args={}
    )
    # Create AdapterSpec based on the RealToxicityPrompts paper: https://arxiv.org/pdf/2009.11462.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=2,  # TODO: why is this so low?
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


def get_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_reasoning_natural_scenario.SRNScenario",
        args={"difficulty": difficulty},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Please solve the following problem.",
        max_train_instances=3,  # limited by the context length
        max_tokens=20,
        input_noun="Rules",
        output_noun="",
    )

    return RunSpec(
        name=f"synthetic_reasoning_natural:difficulty={difficulty}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_srn_metric_specs() + get_generative_harms_metric_specs(),
        groups=["synthetic_reasoning", "synthetic_reasoning_natural"],
    )


def get_gsm_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})
    # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Q: ",
        input_suffix="",
        output_prefix="A: ",
        max_train_instances=5,  # Due to limited context and long example length
        temperature=0.0,
        stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
        max_tokens=400,  # The paper uses 400 tokens as the max sample length
        num_outputs=1,
    )
    return RunSpec(
        name="gsm",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match_indicator"]) + get_generative_harms_metric_specs(),
        groups=["gsm"],
    )


def get_raft_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.raft_scenario.RAFTScenario", args={"subset": subset})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=get_raft_instructions(subset) + "\n",
        input_prefix="",
        output_prefix="Label: ",
        max_train_instances=5,
        temperature=0.0,
        stop_sequences=["\n"],
        num_outputs=1,
        max_tokens=30,  # at most ~50 characters per label
    )

    return RunSpec(
        name=f"raft:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["raft"],
    )


def get_numeracy_spec(
    relation_type: str = "linear", mode: str = "function", seed: str = "0", run_solver: str = "False"
) -> RunSpec:
    run_solver: bool = True if run_solver == "True" else False  # type: ignore
    random_seed = int(seed)
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.numeracy_scenario.NumeracyScenario",
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


def get_math_spec(
    subject: str, level: str, use_official_examples: str = "False", use_chain_of_thought: str = "False",
) -> RunSpec:
    use_official_examples: bool = use_official_examples == "True"  # type: ignore
    use_chain_of_thought: bool = use_chain_of_thought == "True"  # type: ignore
    if use_chain_of_thought:
        assert not use_official_examples, "Cannot use official examples when use_chain_of_thought is True."
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.math_scenario.MATHScenario",
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


def get_boolq_spec(only_contrast=False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.boolq_scenario.BoolQScenario", args={"only_contrast": only_contrast}
    )

    return RunSpec(
        name="boolq" + (":only_contrast=True" if only_contrast else ""),
        scenario_spec=scenario_spec,
        adapter_spec=get_generation_adapter_spec("Passage", "Answer"),
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["boolq"],
    )


def get_lsat_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.lsat_qa_scenario.LSATScenario", args={"task": task})

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        adapter_spec = get_multiple_choice_joint_adapter_spec(
            "The following are multiple choice questions (with answers).", "Passage", "Answer"
        )
        metric_specs = get_basic_metric_specs(["exact_match", "quasi_exact_match"])
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        adapter_spec = get_multiple_choice_separate_adapter_spec(method)
        metric_specs = get_basic_metric_specs([])
    else:
        raise ValueError(f"Invalid adaptation method: {method}")

    return RunSpec(
        name=f"lsat_qa:task={task}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lsat_qa"],
    )


def get_imdb_spec(only_contrast=False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.imdb_scenario.IMDBScenario", args={"only_contrast": only_contrast}
    )

    return RunSpec(
        name="imdb" + (":only_contrast=True" if only_contrast else ""),
        scenario_spec=scenario_spec,
        adapter_spec=get_generation_adapter_spec("Passage", "Sentiment"),
        metric_specs=get_exact_match_metric_specs(),
        groups=["imdb"],
    )


def get_babi_qa_spec(task: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.babi_qa_scenario.BabiQAScenario", args={"task": task})

    return RunSpec(
        name=f"babi_qa:task={task}",
        scenario_spec=scenario_spec,
        # Answers are 1-2 words (1 for all tasks except task 19)
        adapter_spec=get_generation_adapter_spec("Passage", "Answer"),
        metric_specs=get_exact_match_metric_specs(),
        groups=["babi_qa"],
    )


def get_copyright_spec(datatag="pilot", **unused_kwargs) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.copyright_scenario.CopyrightScenario", args=dict(datatag=datatag)
    )

    adapter_spec = get_completion_adapter_spec(temperature=0.2, max_tokens=1024,)

    return RunSpec(
        name=f"copyright:datatag={datatag}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_copyright_metric_specs({"normalize_by_prefix_length": True})
        + get_generative_harms_metric_specs(),
        groups=["copyright"],
    )


def get_disinformation_spec(capability: str = "reiteration", topic: Optional[str] = "covid") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.disinformation_scenario.DisinformationScenario",
        args={"capability": capability, "topic": topic},
    )
    scenario_name = f"disinfo:type={capability}"
    if capability == "reiteration":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="Write headlines that support the thesis.\n",
            input_prefix="Thesis: ",
            output_prefix="Headline:",
            instance_prefix="\n",
            # Justification: Inspection. max_train_instances = 0 or 1 led to worse generations. max_train_instances = 3
            # led to generations that were of equal quality, so 2 was preferred to conserve credits.
            max_train_instances=2,
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in the
            # Pull_Climate_Skepticism.ipynb notebook located at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Amplification/
            temperature=0.7,
            stop_sequences=["\n"],
        )
        metric_specs = get_generative_harms_metric_specs() + get_disinformation_metric_specs(
            args={"name": "reiteration"}
        )
        scenario_name += f",topic={topic}"
    elif capability == "wedging":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            input_prefix="",
            input_suffix="",
            output_prefix="",
            max_train_instances=0,
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in all notebooks at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Wedging/
            temperature=0.7,
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
    if adapter_spec.num_outputs <= 1 and "self_bleu" in {metric_spec.args["name"] for metric_spec in metric_specs}:
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


def get_code_spec(dataset: str, timeout=3) -> RunSpec:
    # `timeout` trades accuracy for time. Used exclusively for APPS. Default from original APPS codebase.
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.code_scenario.CodeScenario", args={"dataset": dataset})

    if dataset == "humaneval":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="",
            # in-context examples are generally too long to fit in the model context window
            max_train_instances=0,
            num_outputs=1,
            temperature=0.2,
            # Taken from the original OpenAI paper to prevent the further generation of irrelevant classes/functions
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint",],
            max_tokens=600,
            input_prefix="",
            input_suffix="",
            output_prefix="",
        )
    else:  # apps.
        # Different in `stop_sequences`.
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="",
            max_train_instances=2,  # Follows the original paper https://arxiv.org/pdf/2105.09938.pdf Appendix D.
            num_outputs=1,
            temperature=0.2,
            stop_sequences=[
                "'''",
                "---",
                '"""',
                "\n\n\n",
            ],  # Manually selected by @lxuechen to prevent the further generation of irrelevant classes/functions
            max_tokens=600,
            input_prefix="",
            input_suffix="",
            output_prefix="",
        )

    return RunSpec(
        name=f"code:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_metric_specs(dataset, timeout) + get_generative_harms_metric_specs(),
        groups=[f"code_{dataset}"],
    )


def get_natural_qa_spec(mode: str) -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={"mode": mode}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Question: " if mode == "closedbook" else "",
        output_prefix="Answer: ",
        max_train_instances=5,
        num_outputs=1,
        max_tokens=300,  # answers are at most 65 words
        temperature=0.0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"natural_qa:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs() + get_generative_harms_metric_specs(),
        groups=["natural_qa", f"natural_qa_{mode}"],
    )


def get_the_pile_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.the_pile_scenario.ThePileScenario", args={"subset": subset}
    )

    return RunSpec(
        name=f"the_pile:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["the_pile"],
    )


def get_ice_spec(**kwargs) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.ice_scenario.ICEScenario", args=kwargs)

    return RunSpec(
        name="ice" + (":" if len(kwargs) > 0 else "") + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())),
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["ice"],
    )


def get_narrativeqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.narrativeqa_scenario.NarrativeQAScenario", args={})

    return RunSpec(
        name="narrative_qa",
        scenario_spec=scenario_spec,
        adapter_spec=get_generation_adapter_spec("Passage", "Answer", max_tokens=100),  # max 30 words
        metric_specs=get_basic_metric_specs(
            ["exact_match", "quasi_exact_match", "f1_score", "rouge-l", "bleu_1", "bleu_4"]
        )
        + get_generative_harms_metric_specs(),
        groups=["narrative_qa"],
    )


def get_synthetic_efficiency_spec(
    num_prompt_tokens: int, num_output_tokens: int, tokenizer: str, random: Optional[str] = None
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario",
        args={"num_prompt_tokens": num_prompt_tokens, "num_instances": 10, "tokenizer": tokenizer},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        max_train_instances=0,
        temperature=0.0,
        stop_sequences=[],
        num_outputs=1,
        max_tokens=num_output_tokens,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        random=random,
    )

    return RunSpec(
        name=f"synthetic_efficiency:tokenizer={tokenizer},num_prompt_tokens={num_prompt_tokens},"
        f"num_output_tokens={num_output_tokens},random={random}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match"]) + get_generative_harms_metric_specs(),
        groups=["synthetic_efficiency"],
    )


def get_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_reasoning_scenario.SyntheticReasoningScenario", args={"mode": mode},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.\n",
        max_train_instances=5,
        temperature=0.0,
        stop_sequences=["\n"],
        num_outputs=1,
        max_tokens=50,  # answer upperbounded by 50 tokens
        input_prefix="",
        input_suffix="",
        output_prefix="| Target: ",
    )
    return RunSpec(
        name=f"synthetic_reasoning:mode={mode}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["synthetic_reasoning", f"synthetic_reasoning_{mode}"],
    )


def get_wikitext_103_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.wikitext_103_scenario.Wikitext103Scenario", args={})

    return RunSpec(
        name="wikitext_103",
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_basic_metric_specs([]),
        groups=["wikitext_103"],
    )


def get_blimp_spec(phenomenon: str, method: str = ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.blimp_scenario.BLiMPScenario", args={"phenomenon": phenomenon}
    )

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        adapter_spec = get_multiple_choice_joint_empty_input_adapter_spec(
            "Please select the grammatical sentence.", "Answer"
        )
        metric_specs = get_basic_metric_specs(["exact_match", "quasi_exact_match"])
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        adapter_spec = get_multiple_choice_separate_adapter_spec(method)
        metric_specs = get_basic_metric_specs([])
    else:
        raise ValueError(f"Invalid adaptation method: {method}")

    return RunSpec(
        name=f"blimp:phenomenon={phenomenon}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["blimp"],
    )


def get_xsum_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "xsum", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512,},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Summarize the given documents.\n",
        input_prefix="Document: ",
        output_prefix="Summary: {",
        # TODO: why } not part of output_suffix?
        max_train_instances=5,
        num_outputs=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"device": device}) + get_generative_harms_metric_specs(),
        groups=["summarization_xsum"],
    )


def get_summarization_adapter_spec(**kwargs) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Summarize the given documents.\n",
        input_prefix="Document: {",
        input_suffix="}\n",
        output_prefix="Summary: {",
        output_suffix="}\n",
        max_train_instances=5,
        num_outputs=1,
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
        **kwargs,
    )


def get_xsum_sampled_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={
            "dataset_name": "xsum-sampled",
            "sampling_min_length": 50,
            "sampling_max_length": 150,
            "doc_max_length": 512,
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"device": device}) + get_generative_harms_metric_specs(),
        groups=["summarization_xsum"],
    )


def get_cnndm_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "cnn-dm", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512,},
    )

    adapter_spec = get_summarization_adapter_spec(
        max_tokens=128,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # From Wu et al. 2021 (https://arxiv.org/pdf/2109.10862.pdf)
    )

    return RunSpec(
        name=f"summarization_cnndm:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"device": device}) + get_generative_harms_metric_specs(),
        groups=["summarization_cnndm"],
    )


def get_empatheticdialogues_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.dialogue_scenarios.EmpatheticDialoguesScenario", args={}
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
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=[],
    )


def get_dyck_language_spec(num_parenthesis_pairs: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.dyck_language_scenario.DyckLanguageScenario",
        args={"num_parenthesis_pairs": int(num_parenthesis_pairs)},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please complete the rest of the following Dyck sequences, "
        "making sure that the parentheses are closed properly.\n",
        input_prefix="Input: ",
        input_suffix="",
        output_prefix="",  # Note: the instance output has a leading space
        output_suffix="",
        temperature=0.0,
        max_train_instances=3,  # Determined by looking at average length of examples to see what fits
        stop_sequences=["\n"],
        max_tokens=5,  # answers are generally at most 1 token due to multiple-choice
        num_outputs=1,
    )

    return RunSpec(
        name=f"dyck_language_np={int(num_parenthesis_pairs)}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match_indicator"]) + get_generative_harms_metric_specs(),
        groups=["dyck_language"],
    )


def get_legal_support_spec(method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.legal_support_scenario.LegalSupportScenario", args={})

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        adapter_spec = get_multiple_choice_joint_adapter_spec(
            "Which statement best supports the passage?",
            "Passage",
            "Answer",
            max_train_instances=3,  # We use 3 because these samples tend to be a bit longer
        )
        metric_specs = get_basic_metric_specs(["exact_match", "quasi_exact_match"])
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        adapter_spec = get_multiple_choice_separate_adapter_spec(method)
        metric_specs = get_basic_metric_specs([])
    else:
        raise ValueError(f"Invalid adaptation method: {method}")

    return RunSpec(
        name="legal_support",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["legal_support"],
    )


def get_entity_matching_spec(dataset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.entity_matching_scenario.EntityMatchingScenario", args={"dataset": dataset}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Are Product A and Product B the same? Yes or No?\n",
        input_prefix="",
        input_suffix="",
        output_prefix=" ",
        max_train_instances=5,
        num_outputs=1,
        max_tokens=5,  # answers are generally 1 token (Yes/No)
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"entity_matching:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["entity_matching"],
    )


def get_entity_data_imputation_spec(dataset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.entity_data_imputation_scenario.EntityDataImputationScenario",
        args={"dataset": dataset},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="What is the missing value?\n",
        input_prefix="",
        input_suffix="",
        output_prefix=" ",
        output_suffix="\n",
        max_train_instances=5,
        num_outputs=1,
        max_tokens=5,
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"entity_data_imputation:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]) + get_generative_harms_metric_specs(),
        groups=["entity_data_imputation"],
    )


@htrack("Extracting adaptation parameters from the BIG-bench task definition and building the RunSpec")
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
                return get_basic_metric_specs(["exact_match", "quasi_exact_match"])
            elif big_bench_metric_name == "exact_str_match":
                metric_names.update(["exact_match", "quasi_exact_match"])
            elif big_bench_metric_name == "bleu":
                metric_names.update(["bleu_1", "bleu_4"])
            elif big_bench_metric_name == "rouge":
                metric_names.update(["rouge-1", "rouge-2", "rouge-l"])
            else:
                hlog(f"Unhandled BIG-bench metric: {big_bench_metric_name}")
                continue

        return get_basic_metric_specs(list(metric_names))

    scenario_spec = ScenarioSpec(
        class_name="benchmark.scenarios.big_bench_scenario.BIGBenchScenario", args={"task": task, "subtask": subtask}
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
        max_train_instances=0,  # Can override with the `MaxTrainInstancesRunExpander`.
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


def get_pubmed_qa_spec(prompt_answer_choices: str) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="benchmark.scenarios.pubmed_qa_scenario.PubMedQAScenario", args={})

    # We are trying to reproduce the Instruct-GPT3's zero-shot performance of 73.2% from
    # "Can large language models reason about medical questions?" (Liévin et al.).
    # Therefore, specify the values of the fields of `AdapterSpec` based on experiment details of the paper.
    # Set `output_prefix` based on Table 1 (titled "Prompt templates") of the paper.
    output_prefix: str = "Answer: "
    if prompt_answer_choices.lower() == "true":
        output_prefix += "among A through C, the answer is "

    # Liévin et al. followed what Kojima et al. did in "Large Language Models are Zero-Shot Reasoners."
    # to extract answers from completions: set the max completion length to a large number and
    # "...pick up the first large letter encountered in the text." Then they set "'Q:'...as a customized stop
    # sequence for all the models except for Instruct-GPT3 to stop the models from repeating questions and
    # answers by themselves." We don't need to do this since our framework has a "multiple_choice_joint"
    # adaptation method that handles the prompt construction for multiple-choice QA for us.
    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        max_train_instances=0,  # We want to reproduce the zero-shot performance.
        # "We sampled one completion per prompt with a temperature of zero..."
        num_outputs=1,
        temperature=0,
        input_prefix="",
        output_prefix=output_prefix,
        # Following the examples in https://vlievin.github.io/medical-reasoning/samples/pubmedqa.html
        reference_prefix="A) ",
    )
    return RunSpec(
        name=f"pubmed_qa:prompt_answer_choices={prompt_answer_choices}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]),
        groups=["pubmed_qa"],
    )


############################################################

CANONICAL_RUN_SPEC_FUNCS: Dict[str, Callable[..., RunSpec]] = {
    "simple1": get_simple1_spec,
    "boolq": get_boolq_spec,
    "imdb": get_imdb_spec,
    "copyright": get_copyright_spec,
    "mmlu": get_mmlu_spec,
    "msmarco": get_msmarco_spec,
    "narrative_qa": get_narrativeqa_spec,
    "commonsense": get_commonsense_spec,
    "lsat_qa": get_lsat_qa_spec,
    "quac": get_quac_spec,
    "wikifact": get_wikifact_spec,
    "babi_qa": get_babi_qa_spec,
    "real_toxicity_prompts": get_real_toxicity_prompts_spec,
    "summarization_xsum": get_xsum_summarization_spec,
    "summarization_xsum_sampled": get_xsum_sampled_summarization_spec,
    "summarization_cnndm": get_cnndm_summarization_spec,
    "truthful_qa": get_truthful_qa_spec,
    "twitter_aae": get_twitter_aae_spec,
    "disinformation": get_disinformation_spec,
    "gsm": get_gsm_spec,
    "math": get_math_spec,
    "natural_qa": get_natural_qa_spec,
    "numeracy": get_numeracy_spec,
    "the_pile": get_the_pile_spec,
    "raft": get_raft_spec,
    "synthetic_efficiency": get_synthetic_efficiency_spec,
    "synthetic_reasoning": get_synthetic_reasoning_spec,
    "synthetic_reasoning_natural": get_synthetic_reasoning_natural_spec,
    "news_qa": get_news_qa_spec,
    "wikitext_103": get_wikitext_103_spec,
    "blimp": get_blimp_spec,
    "code": get_code_spec,
    "empatheticdialogues": get_empatheticdialogues_spec,
    "bold": get_bold_spec,
    "bbq": get_bbq_spec,
    "civil_comments": get_civil_comments_spec,
    "dyck_language": get_dyck_language_spec,
    "legal_support": get_legal_support_spec,
    "entity_matching": get_entity_matching_spec,
    "entity_data_imputation": get_entity_data_imputation_spec,
    "ice": get_ice_spec,
    "big_bench": get_big_bench_spec,
    "pubmed_qa": get_pubmed_qa_spec,
}


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

    return run_specs
