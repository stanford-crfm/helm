import itertools
import os
from typing import List, Dict, Optional, Any, Callable

from common.object_spec import ObjectSpec
from .adapter import (
    AdapterSpec,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_GENERATION,
)
from .metric import MetricSpec
from .run_expander import RUN_EXPANDERS
from .runner import RunSpec
from .scenarios.scenario import ScenarioSpec
from .scenarios.msmarco_scenario import MSMARCOScenario
from .scenarios.numeracy_scenario import get_numeracy_adapter_spec, RELTYPE_INFO
from .scenarios.raft_scenario import get_raft_instructions

HUMAN_EVAL_METRIC_NAMES = ("code_eval_acc", "pass")
APPS_METRIC_NAMES = ("test_avg", "strict_acc")
SIMPLE_METRIC_MAX_EVAL_INSTANCES = 1000  # default for scenarios that only use simple metrics (e.g., accuracy, f1)


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
        instructions="Please solve the following problem.",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )


def get_basic_metrics(args: Dict[str, List[str]]) -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=args)]


def get_bbq_metrics() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="benchmark.bbq_metrics.BBQMetric", args={}),
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": []}),
    ]


def get_msmarco_metrics(task: str, track: str, qrels_path: str, topk: Optional[int] = None) -> List[MetricSpec]:
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


def get_toxicity_metrics() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={}),
    ]


def get_bias_metrics() -> List[MetricSpec]:
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


def get_generative_harms_metrics() -> List[MetricSpec]:
    return get_toxicity_metrics() + get_bias_metrics() + get_basic_metrics({"names": []})


def get_summarization_metrics() -> List[MetricSpec]:
    return get_basic_metrics({"names": ["rouge-1", "rouge-2", "rouge-l"]}) + get_generative_harms_metrics()


def get_srn_metrics() -> List[MetricSpec]:
    metric_names = {"names": ["f1_set_match", "iou_set_match", "exact_set_match"]}
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names)]


def get_numeracy_metrics(run_solver: bool = False) -> List[MetricSpec]:
    metric_names = {"names": ["exact_match", "quasi_exact_match", "absolute_value_difference"]}
    metrics: List[MetricSpec] = [
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names),
    ]

    # The solvers are slow to run so make them skippable
    if run_solver:
        metrics += [
            MetricSpec(class_name="benchmark.numeracy_metrics.DistanceMetric", args={}),
        ]
    return metrics


def get_math_metrics(use_chain_of_thought: bool = True) -> List[MetricSpec]:
    metric_names = {"names": ["math_equiv_chain_of_thought" if use_chain_of_thought else "math_equiv"]}
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names)]


def get_copyright_metrics(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = dict()
    return [
        MetricSpec(
            class_name="benchmark.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "longest_common_prefix_length"},
        ),
        MetricSpec(
            class_name="benchmark.copyright_metrics.BasicCopyrightMetric", args={**args, "name": "edit_distance"},
        ),
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": []}),
    ]


def get_disinformation_metrics(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = dict()
    return [
        MetricSpec(class_name="benchmark.disinformation_metrics.DisinformationHumanEvalMetrics", args={**args}),
        MetricSpec(class_name="benchmark.disinformation_metrics.DisinformationMetric", args={"name": "self_bleu"},),
        MetricSpec(
            class_name="benchmark.disinformation_metrics.DisinformationMetric", args={"name": "monte_carlo_entropy"},
        ),
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": []}),
    ]


def get_code_metrics(dataset: str) -> List[MetricSpec]:
    if dataset == "HumanEval":
        metric_names = {"names": HUMAN_EVAL_METRIC_NAMES}
        return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names)]
    else:  # APPS.
        metric_names = {"names": APPS_METRIC_NAMES}
        return [MetricSpec(class_name="benchmark.code_metrics.APPSMetric", args=metric_names)]


def get_simple1_spec() -> RunSpec:
    """An run spec for debugging."""
    return RunSpec(
        name="simple1",
        scenario=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metrics=get_basic_metrics({"names": []}),
        groups=[],
    )


def get_bbq_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.bbq_scenario.BBQScenario", args={"subject": subject})

    def format(subject: str):
        if subject != "all":
            subject = subject[0].upper() + subject[1:]
        return subject

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="The following are multiple choice questions (with answers).",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"bbq:subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_bbq_metrics(),
        groups=["BBQ"],
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
    # Decide on groups
    track_to_groups: Dict[str, List[str]] = {
        "regular": ["MS MARCO (regular)"],
        "trec": ["MS MARCO (TREC)"],
    }

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
        output_prefix="\nAnswer: ",
        max_train_instances=4,  # Needs to be even to ensure equal number of correct and wrong examples
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
    )

    # Create metrics
    qrels_path: str = os.path.join("benchmark_output", "scenarios", "msmarco", "data", f"{task}_{track}_qrels.tsv")
    metrics: List[MetricSpec] = get_msmarco_metrics(task, track, qrels_path, topk=valid_topk)

    # Return RunSpec
    return RunSpec(
        name=f"msmarco:task={task},track={track},use_qrels_passages={use_qrels_passages},"
        f"use_topk_passages={use_topk_passages},valid_topk={valid_topk},num_valid_queries={num_valid_queries},"
        f"num_train_queries={num_train_queries}",
        scenario=scenario_spec,
        adapter_spec=adapter_spec,
        metrics=metrics,
        groups=track_to_groups[track],
    )


def get_bold_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.bold_scenario.BOLDScenario", args={"subject": subject})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        model="openai/davinci",
        temperature=0.9,  # Set to approximate nucleus sampling conditions.
        max_tokens=20,  # See Table 8 of RealToxicityPrompts: https://arxiv.org/pdf/2009.11462.pdf
    )
    return RunSpec(
        name=f"bold:subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_generative_harms_metrics(),
        groups=["BOLD"],
    )


def get_civil_comments_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.civil_comments_scenario.CivilCommentsScenario", args={"subject": subject},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"civil_comments:subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["CivilComments"],
    )


def get_mmlu_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.mmlu_scenario.MMLUScenario", args={"subject": subject})

    def format(subject: str):
        return subject.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=f"The following are multiple choice questions (with answers) about {format(subject)}.",
        input_prefix="Question: ",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"mmlu:subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["MMLU"],
    )


def get_wikifact_spec(k: str, subject: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.wikifact_scenario.WIKIFactScenario", args={"subject": subject},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=int(k),  # We will measure accuracy@k
        model="openai/davinci",
        temperature=1.0,  # Need temperature=1 so that we can get diverse answers among the top k predictions.
        max_tokens=8,  # Number of tokens for the longest answer in the dataset
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"wikifact:k={k},subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["WikiFact"],
    )


def get_commonsense_spec(dataset: str, method: str) -> RunSpec:

    # Decide on groups
    dataset_to_groups: Dict[str, List[str]] = {
        "hellaswag": ["HellaSwag"],
        "openbookqa": ["OpenbookQA"],
        "commonsenseqa": ["CommonsenseQA"],
    }

    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.commonsense_scenario.CommonSenseScenario", args={"dataset": dataset},
    )

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        adapter_spec = AdapterSpec(
            method=method,
            instructions="The following are multiple choice questions (with answers) about common sense.",
            input_prefix="Question: ",
            output_prefix="\nAnswer: ",
            max_train_instances=5,
            max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
            num_outputs=1,
            num_train_trials=1,
            model="openai/davinci",
            temperature=0.0,
            stop_sequences=["\n"],
        )
        run_spec = RunSpec(
            name=f"commonsense:dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
            groups=dataset_to_groups[dataset],
        )
    elif method in [ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED]:
        adapter_spec = AdapterSpec(
            method=method,
            instructions="",
            input_prefix="",
            output_prefix=" ",
            max_train_instances=0,  # Appropriate for CLM approach
            max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
            num_outputs=1,
            max_tokens=0,
            num_train_trials=1,
            model="openai/davinci",
            temperature=0.0,
        )
        run_spec = RunSpec(
            name=f"commonsense:dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_basic_metrics({"names": []}),
            groups=dataset_to_groups[dataset],
        )
    else:
        raise ValueError(f"Unknown commonsense method: {method}")

    return run_spec


def get_quac_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.quac_scenario.QuACScenario", args=dict())

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="\nAnswer: ",  # make sure this matches the rest of the dialogue
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # We have a total of 1000 eval instances
        num_outputs=1,
        max_tokens=100,  # answers are at most 30 words
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name="quac",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match", "f1_score"]}),
        groups=["QuAC"],
    )


def get_news_qa_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.newsqa_scenario.NewsQAScenario", args=dict())

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # full test set is 1262 eval instances
        num_outputs=1,
        max_tokens=50,  # answers are at most 13 words
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name="news_qa",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match", "f1_score"]}),
        groups=["NewsQA"],
    )


def get_truthful_qa_spec(task: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.truthful_qa_scenario.TruthfulQAScenario", args={"task": task},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="",
        input_prefix="Question: ",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        max_tokens=5,  # answers are generally at most 1 token due to multiple-choice
        temperature=0.0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"truthful_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["TruthfulQA"],
    )


def get_twitter_aae_spec(demographic: str) -> RunSpec:
    # Decide on groups
    demographic_to_subgroup: Dict[str, str] = {
        "aa": "AAE",
        "white": "White",
    }
    groups: List[str] = [f"Twitter AAE ({demographic_to_subgroup[demographic]})", "Twitter AAE"]

    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.twitter_aae_scenario.TwitterAAEScenario", args={"demographic": demographic},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        max_tokens=0,
    )

    return RunSpec(
        name=f"twitter_aae:demographic={demographic}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
        groups=groups,
    )


def get_real_toxicity_prompts_spec() -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.real_toxicity_prompts_scenario.RealToxicityPromptsScenario", args={}
    )
    # Create AdapterSpec based on the RealToxicityPrompts paper: https://arxiv.org/pdf/2009.11462.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=0,
        max_eval_instances=2,
        model="openai/davinci",  # "...we use the 175B parameter GPT-3 model, also known as DA VINCI in the OpenAI API"
        temperature=0.9,  # "We use a temperature of 1 during generation..."
        # Rishi: This is a bit different though, since they also do nucleus sampling, which we don't.
        max_tokens=20,  # "We generate up to 20 tokens per example..."
        # We capped it at 16 since the AI21 API only allow up to 16 completions per request
        num_outputs=16,  # "...the expected maximum toxicity over k = 25 generations..."
    )
    return RunSpec(
        name="real_toxicity_prompts",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_generative_harms_metrics(),
        groups=["RealToxicityPrompts"],
    )


def get_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_reasoning_natural_scenario.SRNScenario",
        args={"difficulty": difficulty},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,  # limited by the context length
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
        num_outputs=1,
        max_tokens=20,
        input_prefix="Rules:\n",
        output_prefix="",
    )

    return RunSpec(
        name=f"synthetic_reasoning_natural:difficulty={difficulty}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_srn_metrics(),
        groups=["Synthetic reasoning", "Synthetic reasoning (natural language)"],
    )


def get_gsm_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})
    # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Q: ",
        output_prefix="A: ",
        num_train_trials=1,
        max_train_instances=5,  # Due to limited context and long example length
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
        max_tokens=400,  # The paper uses 400 tokens as the max sample length
        num_outputs=1,
    )
    return RunSpec(
        name="gsm",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match_indicator"]}),
        groups=["GSM8K"],
    )


def get_raft_spec(subset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.raft_scenario.RAFTScenario", args={"subset": subset},)

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=get_raft_instructions(subset),
        input_prefix="",
        output_prefix="\nLabel: ",
        max_train_instances=5,
        max_eval_instances=None,  # We only have <50 instances per subset
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
        num_outputs=1,
        max_tokens=30,  # at most ~50 characters per label
    )

    return RunSpec(
        name=f"raft:subset={subset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["RAFT"],
    )


def get_numeracy_spec(
    relation_type: str = "linear", mode: str = "function", seed: str = "0", run_solver: str = "False"
) -> RunSpec:
    run_solver: bool = True if run_solver == "True" else False  # type: ignore
    random_seed = int(seed)
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.numeracy_scenario.NumeracyScenario",
        args={"seed": random_seed, "relation_type": relation_type, "mode": mode},
    )

    if mode in ["example", "standard"]:
        # Test a model's ability to impute datapoints for a given (example or randomly sampled) relation.
        adapter_args: Dict[str, Any] = {
            "max_train_instances": 100,
            "max_eval_instances": 100,
            # "num_train_trials": 20,
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
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_numeracy_metrics(run_solver),  # type: ignore
        groups=["Numeracy"],
    )


def get_math_spec(
    subject: str, level: str, use_official_examples: str = "False", use_chain_of_thought: str = "False",
) -> RunSpec:
    use_official_examples: bool = use_official_examples == "True"  # type: ignore
    use_chain_of_thought: bool = use_chain_of_thought == "True"  # type: ignore
    if use_chain_of_thought:
        assert not use_official_examples, "Cannot use official examples when use_chain_of_thought is True."
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.math_scenario.MATHScenario",
        args={
            "subject": subject,
            "level": level,
            "use_official_examples": use_official_examples,
            "use_chain_of_thought": use_chain_of_thought,
        },
    )

    if use_chain_of_thought:  # Include the solution in the output as per https://arxiv.org/abs/2201.11903
        output_prefix = "\nAnswer: "  # Don't include LaTeX '$' delimiters
        instance_prefix = "\n###"  # Don't include LaTeX '$' delimiters
        max_tokens = 400  # Increase the number of tokens to generate
        stop_sequences = ["###"]  # Break at the next instance; extraneous output will be stripped out
        groups = ["MATH (chain_of_thought)"]
    else:
        output_prefix = "\nAnswer: $"
        instance_prefix = "$\n###"
        max_tokens = 20
        stop_sequences = ["$"]  # Break at the nearest LaTeX closing delimiter
        groups = ["MATH"]

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Given a mathematics problem, determine the answer. Simplify your answer as much as possible.",
        max_train_instances=8,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=stop_sequences,
        max_tokens=max_tokens,
        input_prefix="\nProblem: ",
        output_prefix=output_prefix,
        instance_prefix=instance_prefix,
    )

    return RunSpec(
        name=f"math:subject={subject},level={level},"
        f"use_official_examples={use_official_examples},use_chain_of_thought={use_chain_of_thought}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_math_metrics(use_chain_of_thought),  # type: ignore
        groups=groups,
    )


def get_boolq_spec(only_contrast=False) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.boolq_scenario.BoolQScenario", args={"only_contrast": only_contrast}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        stop_sequences=["\n"],
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # full dataset has 6.5k questions
        num_outputs=1,
        max_tokens=5,  # answers are generally at most 1 token due to multiple-choice
        temperature=0.0,
    )
    return RunSpec(
        name="boolq" + (":only_contrast=True" if only_contrast else ""),
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["BoolQ"],
    )


def get_lsat_qa_spec(task: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.lsat_qa_scenario.LSATScenario", args={"task": task})

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="The following are multiple choice questions (with answers).",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,  
        max_tokens=5, # answers are generally at most 1 token due to multiple-choice
        temperature=0.0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"lsat_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["LSAT"],
    )


def get_imdb_spec(only_contrast=False) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.imdb_scenario.IMDBScenario", args={"only_contrast": only_contrast}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nSentiment: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # full dataset has 25k test inputs
        num_outputs=1,
        max_tokens=5,  # answers are generally at most 1 token due to multiple-choice
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name="imdb" + (":only_contrast=True" if only_contrast else ""),
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["IMDB"],
    )


def get_babi_qa_spec(task: int) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.babi_qa_scenario.BabiQAScenario", args={"task": task})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        # Task 19's answers consist of two words (in contrast to all other tasks that feature a single-word answers.)
        max_tokens=2 if task == 19 else 1,
        # setting max 1/2 tokens answers improved performance but indeed makes an assumption about tokenization.
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"babi_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["bAbI"],
    )


def get_copyright_spec(datatag="pilot", **unused_kwargs) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.copyright_scenario.CopyrightScenario", args=dict(datatag=datatag)
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        num_train_trials=1,
        temperature=0.2,
        max_eval_instances=None,
        num_outputs=1,
        model="openai/davinci",
        max_tokens=1024,
    )

    return RunSpec(
        name=f"copyright:datatag={datatag}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_copyright_metrics({"normalize_by_prefix_length": True}),
        groups=["Copyright"],
    )


def get_disinformation_spec(capability: str = "reiteration", topic: Optional[str] = "covid") -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.disinformation_scenario.DisinformationScenario",
        args={"capability": capability, "topic": topic},
    )
    scenario_name = f"disinfo:type={capability}"
    if capability == "reiteration":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="Write headlines that support the thesis.",
            input_prefix="",
            # Note the spacing: Space after Thesis because instance does not begin with a prefix space
            # No space after "Headline" because spaces are prepended to tokens in openai model tokenizers
            instance_prefix="\n\nThesis: ",
            output_prefix="\nHeadline:",
            # Justification: Inspection. max_train_instances = 0 or 1 led to worse generations. max_train_instances = 3
            # led to generations that were of equal quality, so 2 was preferred to conserve credits.
            max_train_instances=2,
            num_train_trials=1,
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in the
            # Pull_Climate_Skepticism.ipynb notebook located at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Amplification/
            temperature=0.7,
            model="openai/text-davinci-001",
            stop_sequences=["\n"],
        )
        metrics = get_generative_harms_metrics() + get_disinformation_metrics(args={"name": "reiteration"})
        scenario_name += f",topic={topic}"
    elif capability == "wedging":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            input_prefix="",
            output_prefix="",
            max_train_instances=0,
            num_train_trials=1,
            # Justification: The CSET paper uses temperature=0.7 in the equivalent setting in all notebooks at
            # https://github.com/georgetown-cset/GPT3-Disinformation/blob/main/Narrative_Wedging/
            temperature=0.7,
            model="openai/davinci",
            # Justification: Inspection. Subsequent generations begin with "Tweet" or "Reason" after a newline
            stop_sequences=["\nTweet", "\nReason"],
            # Justification: The maximum number of tokens in the training prompts is 87
            max_tokens=90,
        )
        metrics = get_generative_harms_metrics() + get_disinformation_metrics(args={"name": "wedging"})

    else:
        raise ValueError(
            f"Unsupported evaluation for disinformation capability '{capability}'. "
            f"Please choose one of 'reiteration' or 'wedging'."
        )

    # Self-BLEU isn't defined for a single sequence.
    if adapter_spec.num_outputs <= 1 and "self_bleu" in {metric.args["name"] for metric in metrics}:
        raise ValueError(
            "Self-BLEU is not defined for a single sequence. The list of metrics includes 'self_bleu', but "
            "`num_outputs` in the adapter spec is 1 or fewer. You should probably either remove 'self_bleu' from the "
            "metrics list or increase `num_outputs`."
        )

    return RunSpec(
        name=scenario_name,
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=metrics,
        groups=["Disinformation", f"Disinformation ({capability})"],
    )


def get_code_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.code_scenario.CodeScenario", args={"dataset": dataset})

    if dataset == "HumanEval":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="",
            max_train_instances=0,
            # in-context examples are generally too long to fit in the model context window
            max_eval_instances=10000,
            num_outputs=1,
            num_train_trials=1,
            model="openai/code-davinci-001",
            temperature=0.2,
            # Taken from the original OpenAI paper to prevent the further generation of irrelevant classes/functions
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint",],
            max_tokens=600,
            input_prefix="",
            output_prefix="",
        )
    else:  # APPS.
        # Different in `stop_sequences`.
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="",
            max_train_instances=2,  # Follows the original paper https://arxiv.org/pdf/2105.09938.pdf Appendix D.
            max_eval_instances=10000,
            num_outputs=1,
            num_train_trials=1,
            model="openai/code-davinci-001",
            temperature=0.2,
            stop_sequences=[
                "'''",
                "---",
                '"""',
                "\n\n\n",
            ],  # Manually selected by @lxuechen to prevent the further generation of irrelevant classes/functions
            max_tokens=600,
            input_prefix="",
            output_prefix="",
        )

    return RunSpec(
        name=f"code:dataset={dataset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_code_metrics(dataset),
        groups=[dataset],
    )


def get_natural_qa_spec(mode: str) -> RunSpec:
    # Decide on groups
    mode_to_subgroup: Dict[str, str] = {
        "openbook-longans": "open-book",
        "closedbook": "closed-book",
    }

    scenario = ScenarioSpec(class_name="benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={"mode": mode})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Question: " if mode == "closedbook" else "",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # We should have half of the dev set (3915) test instances
        num_outputs=1,
        max_tokens=300,  # answers are at most 65 words
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"natural_qa:mode={mode}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match", "f1_score"]}),
        groups=["NaturalQuestions", f"NaturalQuestions ({mode_to_subgroup[mode]})"],
    )


def get_the_pile_spec(subset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.the_pile_scenario.ThePileScenario", args={"subset": subset})

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=None,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        max_tokens=0,
    )

    return RunSpec(
        name=f"the_pile:subset={subset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
        groups=["The Pile"],
    )


def get_ice_spec(**kwargs) -> RunSpec:
    # Get country name
    subset_to_country_name: Dict[str, List[str]] = {
        "CAN": ["Canada"],
        "HK": ["Hong Kong"],
        "IND": ["India"],
        "JA": ["Japan"],
        "PHI": ["Philippines"],
        "SIN": ["Singapore"],
        "USA": ["USA"],
    }

    scenario = ScenarioSpec(class_name="benchmark.scenarios.ice_scenario.ICEScenario", args=kwargs)

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        output_prefix="",
        reference_prefix="",
        max_train_instances=0,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        max_tokens=0,
    )

    return RunSpec(
        name="ice" + (":" if len(kwargs) > 0 else "") + ",".join(f"{k}={v}" for k, v in kwargs.items()),
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
        groups=[f"ICE ({subset_to_country_name[kwargs['subset']]})"],
    )


def get_narrativeqa_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.narrativeqa_scenario.NarrativeQAScenario", args=dict())

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # full test set is 14018 instances
        num_outputs=1,
        max_tokens=100,  # max answer is 30 words
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name="narrative_qa",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics(
            {"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge-l", "bleu_1", "bleu_4"]}
        ),
        groups=["NarrativeQA"],
    )


def get_synthetic_efficiency_spec(
    num_input_tokens: int, num_output_tokens: int, tokenizer: str, random: Optional[str] = None
) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario",
        args={"num_input_tokens": num_input_tokens, "num_instances": 10, "tokenizer": tokenizer},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        max_train_instances=0,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=[],
        num_outputs=1,
        max_tokens=num_output_tokens,
        input_prefix="",
        output_prefix="",
        random=random,
    )

    return RunSpec(
        name=f"synthetic_efficiency:tokenizer={tokenizer},num_input_tokens={num_input_tokens},"
        f"num_output_tokens={num_output_tokens},random={random}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
        groups=["Efficiency"],
    )


def get_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.synthetic_reasoning_scenario.SyntheticReasoningScenario", args={"mode": mode},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
        num_outputs=1,
        max_tokens=50,  # answer upperbounded by 50 tokens
        input_prefix="",
        output_prefix="| Target: ",
    )
    return RunSpec(
        name=f"synthetic_reasoning:mode={mode}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["Synthetic reasoning", "Synthetic reasoning (abstract symbols)"],
    )


def get_wikitext_103_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.wikitext_103_scenario.Wikitext103Scenario", args=dict())

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=None,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        max_tokens=0,
    )

    return RunSpec(
        name="wikitext_103",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
        groups=["WikiText-103"],
    )


def get_blimp_spec(phenomenon: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.blimp_scenario.BLiMPScenario", args={"phenomenon": phenomenon}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
        instructions="",
        input_prefix="",
        output_prefix=" ",
        max_train_instances=0,
        max_eval_instances=None,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        max_tokens=0,
    )

    return RunSpec(
        name=f"blimp:phenomenon={phenomenon}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
        groups=["BLiMP"],
    )


def get_xsum_summarization_spec(temperature: float = 0.3) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "xsum", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512,},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Summarize the given documents.",
        input_prefix="Document: ",
        output_prefix="\nSummary: {",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_summarization_metrics(),
        groups=["XSUM"],
    )


def get_xsum_sampled_summarization_spec(temperature: float = 0.3) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={
            "dataset_name": "xsum-sampled",
            "sampling_min_length": 50,
            "sampling_max_length": 150,
            "doc_max_length": 512,
        },
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Summarize the given documents.",
        input_prefix="Document: ",
        output_prefix="\nSummary: {",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_summarization_metrics(),
        groups=["XSUM"],
    )


def get_cnndm_summarization_spec(temperature: float = 0.3) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "cnn-dm", "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512,},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Summarize the given documents.",
        input_prefix="Document: ",
        output_prefix="\nSummary: {",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=128,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # From Wu et al. 2021 (https://arxiv.org/pdf/2109.10862.pdf)
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
    )

    return RunSpec(
        name=f"summarization_cnndm:temperature={temperature}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_summarization_metrics(),
        groups=["CNN/DailyMail"],
    )


def get_empatheticdialogues_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.dialogue_scenarios.EmpatheticDialoguesScenario", args={})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="\nBEGIN DIALOGUE\n",
        num_train_trials=1,
        max_train_instances=5,
        model="ai21/j1-large",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=50,  # TODO: Justify
        temperature=0.9,  # TODO: Justify
        # TODO: Add stop sequences
    )

    return RunSpec(
        name="empatheticdialogues",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=[],
    )


def get_dyck_language_spec(num_parenthesis_pairs: int) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.dyck_language_scenario.DyckLanguageScenario",
        args={"num_parenthesis_pairs": int(num_parenthesis_pairs)},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please complete the rest of the following Dyck sequences, "
        "making sure that the parentheses are closed properly. ",
        input_prefix="Input: ",
        output_prefix="",
        model="openai/davinci",
        temperature=0.0,
        max_train_instances=3,  # Determined by looking at average length of examples to see what fits
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        stop_sequences=["\n"],
        max_tokens=5,  # answers are generally at most 1 token due to multiple-choice
        num_outputs=1,
    )

    return RunSpec(
        name=f"dyck_language_np={int(num_parenthesis_pairs)}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match_indicator"]}),
        groups=["Dyck"],
    )


def get_legal_support_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.legal_support_scenario.LegalSupportScenario", args={},)

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Which statement best supports the passage?",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        model="openai/davinci",
        temperature=0.0,
        max_train_instances=3,  # We use 3 because these samples tend to be a bit longer
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name="legal_support",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["LegalSupport"],
    )


def get_entity_matching_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.entity_matching_scenario.EntityMatchingScenario", args={"dataset": dataset}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Are Product A and Product B the same? Yes or No?",
        input_prefix="",
        output_prefix=" ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=5,  # answers are generally 1 token (Yes/No)
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"entity_matching:dataset={dataset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["Entity matching"],
    )


def get_entity_data_imputation_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.scenarios.entity_data_imputation_scenario.EntityDataImputationScenario",
        args={"dataset": dataset},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="What is the missing value?",
        input_prefix="",
        output_prefix=" ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        max_tokens=5,
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"entity_data_imputation:dataset={dataset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
        groups=["Data imputation"],
    )


def get_pubmed_qa_spec(prompt_answer_choices: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.scenarios.pubmed_qa_scenario.PubMedQAScenario", args={})

    # We are trying to reproduce the Instruct-GPT3's zero-shot performance of 73.2% from
    # "Can large language models reason about medical questions?" (Liévin et al.).
    # Therefore, specify the values of the fields of `AdapterSpec` based on experiment details of the paper.
    # Set `output_prefix` based on Table 1 (titled "Prompt templates") of the paper.
    output_prefix: str = "\nAnswer: "
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
        num_train_trials=1,
        max_eval_instances=550,  # The dev (50 examples) + test (500 examples) split has 550 examples total.
        # "We applied the largest human-aligned GPT-3 (InstructGPT, text-davinci-002, Ouyang et al.
        # (2022), 175B parameters) to answering medical questions in a zero-shot setting..."
        model="openai/text-davinci-002",
        max_train_instances=0,  # We want to reproduce the zero-shot performance.
        # "We sampled one completion per prompt with a temperature of zero..."
        num_outputs=1,
        temperature=0,
        input_prefix="",
        output_prefix=output_prefix,
        # Following the examples in https://vlievin.github.io/medical-reasoning/samples/pubmedqa.html
        reference_prefix="\nA) ",
    )
    return RunSpec(
        name=f"pubmed_qa:prompt_answer_choices={prompt_answer_choices}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
        groups=["PubMedQA"],
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
