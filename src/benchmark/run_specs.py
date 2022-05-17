from typing import List, Dict, Optional, Any, Callable
from common.object_spec import ObjectSpec
from .adapter import (
    AdapterSpec,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE,
    ADAPT_GENERATION,
    ADAPT_LANGUAGE_MODELING_MINIMAL_PAIRS,
)
from .commonsense_qa_scenario import MULTI_CHOICE_QUESTION_ANSWERING_METHOD, CAUSAL_LANGUAGE_MODELING_METHOD
from .math_scenario import OFFICIAL_MATH_INSTRUCTIONS, OFFICIAL_MATH_PROMPT
from .metric import MetricSpec
from .numeracy_scenario import get_numeracy_adapter_spec, RELTYPE_INFO
from .raft_scenario import get_raft_instructions
from .run_expander import RUN_EXPANDERS
from .runner import RunSpec
from .scenario import ScenarioSpec

HUMAN_EVAL_METRIC_NAMES = ("code_eval_acc", "pass")
APPS_METRIC_NAMES = ("test_avg", "strict_acc")
SIMPLE_METRIC_MAX_EVAL_INSTANCES = 1000  # default for scenarios that only use simple metrics (e.g., accuracy, f1)


def get_scenario_spec1() -> ScenarioSpec:
    return ScenarioSpec(
        class_name="benchmark.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )


def get_scenario_spec_tiny():
    return ScenarioSpec(
        class_name="benchmark.simple_scenarios.Simple1Scenario",
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
    return [MetricSpec(class_name="benchmark.bbq_metrics.BBQMetric", args={})]


def get_commonsense_qa_metrics(args: Dict[str, Any]) -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.commonsense_qa_metrics.CommonSenseQAMetric", args=args)]


def get_msmarco_metrics() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="benchmark.msmarco_metrics.MSMARCOMetric",
            args={"name": "mean_reciprocal_rank", "topk_list": [10]},
        ),
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args={"names": []}),
    ]


def get_toxicity_metrics() -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={})]


def get_srn_metrics() -> List[MetricSpec]:
    metric_names = {"names": ["f1_set_match", "iou_set_match", "exact_set_match"]}
    return [MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names)]


def get_numeracy_metrics(relation_type: str, run_solver: bool = True) -> List[MetricSpec]:
    metric_names = {"names": ["exact_match", "quasi_exact_match", "absolute_value_difference"]}
    metrics = [
        MetricSpec(class_name="benchmark.basic_metrics.BasicMetric", args=metric_names),
    ]
    if (
        relation_type not in ["parabola", "paraboloid"] or run_solver
    ):  # the solvers are slow to run so make them skippable
        metrics += [
            MetricSpec(class_name="benchmark.numeracy_metrics.DistanceMetric", args={}),
        ]
    return metrics


def get_math_metrics() -> List[MetricSpec]:
    metric_names = {"names": ["math_equiv"]}
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
    ]


def get_disinformation_metrics(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = dict()
    return [
        MetricSpec(
            class_name="benchmark.disinformation_metrics.DisinformationMetric", args={**args, "name": "self_bleu"},
        ),
        MetricSpec(
            class_name="benchmark.disinformation_metrics.DisinformationMetric",
            args={**args, "name": "monte_carlo_entropy"},
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
    )


def get_bbq_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.bbq_scenario.BBQScenario", args={"subject": subject})

    def format(subject: str):
        if subject != "all":
            subject = subject[0].upper() + subject[1:]
        return subject

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
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

    return RunSpec(name="bbq", scenario=scenario, adapter_spec=adapter_spec, metrics=get_bbq_metrics())


def get_msmarco_spec(
    task: str, track: str = "regular", topk: str = "30", num_eval_queries: str = "500", num_train_queries: str = "1000"
) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.msmarco_scenario.MSMARCOScenario",
        args={
            "task": task,
            "track": track,
            "topk": int(topk),
            "num_eval_queries": int(num_eval_queries),
            "num_train_queries": int(num_train_queries),
        },
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        instructions="",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        max_train_instances=4,  # TODO: @Dilara - Justify
        max_eval_instances=200,  # TODO: @Dilara - Justify
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"msmarco:task={task},track={track},topk={topk},num_eval_queries={num_eval_queries},"
        f"num_train_queries={num_train_queries}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_msmarco_metrics(),
    )


def get_bold_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.bold_scenario.BOLDScenario", args={"subject": subject})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,  # TODO: Rishi - setting for 1 to monitor token use; @Ryan please follow-up on this
        model="openai/davinci",
        temperature=0.9,  # TODO: Setting based on conversation with John Hewitt; @Ryan please better justify
        max_tokens=20,  # See Table 8 of RealToxicityPrompts: https://arxiv.org/pdf/2009.11462.pdf
    )
    return RunSpec(name="bold", scenario=scenario, adapter_spec=adapter_spec, metrics=get_toxicity_metrics())


def get_civil_comments_spec(subject: str, data_path: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.civil_comments_scenario.CivilCommentsScenario",
        args={"subject": subject, "data_path": data_path},
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
        name="civil_comments",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_mmlu_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.mmlu_scenario.MMLUScenario", args={"subject": subject})

    def format(subject: str):
        return subject.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
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
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_wikifact_spec(k: str, subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.wikifact_scenario.WIKIFactScenario", args={"subject": subject},)

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
    )


def get_commonsense_qa_spec(dataset: str, method: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.commonsense_qa_scenario.CommonSenseQAScenario",
        args={"dataset": dataset, "method": method,},
    )

    if method == MULTI_CHOICE_QUESTION_ANSWERING_METHOD:
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE,
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
            name=f"commonsense_qa:dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_basic_metrics({"names": ["exact_match"]}),
        )
    elif method == CAUSAL_LANGUAGE_MODELING_METHOD:
        n_choice = {"hellaswag": 4, "openbookqa": 4, "commonsenseqa": 5, "piqa": 2, "siqa": 3,}[dataset]
        adapter_spec = AdapterSpec(
            method=ADAPT_LANGUAGE_MODELING,
            instructions="",
            input_prefix="",
            output_prefix="",
            max_train_instances=0,  # Appropriate for CLM approach
            max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES * n_choice * 2,
            num_outputs=1,
            max_tokens=0,
            num_train_trials=1,
            model="openai/davinci",
            temperature=0.0,
        )
        run_spec = RunSpec(
            name=f"commonsense_qa:dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_commonsense_qa_metrics({"n_choice": n_choice}),
        )
    else:
        raise ValueError(f"Unknown commonsense QA method: {method}")

    return run_spec


def get_quac_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.quac_scenario.QuACScenario", args=dict())

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
    )


def get_news_qa_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.newsqa_scenario.NewsQAScenario", args=dict())

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
    )


def get_truthful_qa_spec(task: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.truthful_qa_scenario.TruthfulQAScenario", args={"task": task},)

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        instructions="",
        input_prefix="Question: ",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        max_tokens=5,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    return RunSpec(
        name=f"truthful_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_twitter_aae_spec(demographic: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.twitter_aae_scenario.TwitterAAEScenario", args={"demographic": demographic},
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
    )


def get_real_toxicity_prompts_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.real_toxicity_prompts_scenario.RealToxicityPromptsScenario", args={})
    # Create AdapterSpec based on the RealToxicityPrompts paper: https://arxiv.org/pdf/2009.11462.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=0,
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        model="openai/davinci",  # "...we use the 175B parameter GPT-3 model, also known as DA VINCI in the OpenAI API"
        temperature=0.9,  # "We use a temperature of 1 during generation..."
        # Rishi: This is a bit different though, since they also do nucleus sampling, which we don't.
        max_tokens=20,  # "We generate up to 20 tokens per example..."
        # We capped it at 16 since the AI21 API only allow up to 16 completions per request
        num_outputs=16,  # "...the expected maximum toxicity over k = 25 generations..."
    )
    return RunSpec(
        name="real_toxicity_prompts", scenario=scenario, adapter_spec=adapter_spec, metrics=get_toxicity_metrics(),
    )


def get_synthetic_reasoning_natural_spec(difficulty: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.synthetic_reasoning_natural_scenario.SRNScenario", args={"difficulty": difficulty}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,  # TODO: @Tony W. - Justify
        max_eval_instances=100,  # TODO: @Tony W. - Justify
        num_outputs=3,  # TODO: @Tony W. - Justify
        num_train_trials=1,
        model="openai/davinci",
        temperature=1.0,  # TODO: @Tony W. - Justify; should it be 0?
        stop_sequences=["\n"],
        max_tokens=20,
        input_prefix="Rules:\n",
        output_prefix="",
    )

    return RunSpec(
        name=f"synthetic_reasoning_natural:difficulty={difficulty}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_srn_metrics(),
    )


def get_gsm_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.gsm_scenario.GSM8KScenario", args={})
    # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=3,  # Due to limited context and long example length
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
    )


def get_raft_spec(subset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.raft_scenario.RAFTScenario", args={"subset": subset},)

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
        max_tokens=30,  # at most ~50 characters per label
    )

    return RunSpec(
        name=f"raft:subset={subset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_numeracy_spec(
    relation_type: str = "linear", mode: str = "function", seed: str = "0", run_solver: bool = True
) -> RunSpec:
    random_seed = int(seed)
    scenario = ScenarioSpec(
        class_name="benchmark.numeracy_scenario.NumeracyScenario",
        args={"seed": random_seed, "relation_type": relation_type, "mode": mode,},
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
    adapter_spec = get_numeracy_adapter_spec(**adapter_args)

    return RunSpec(
        name=f"numeracy:relation_type={relation_type},mode={mode}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_numeracy_metrics(relation_type, run_solver=run_solver),
    )


def get_math_spec(subject: str, level: str, use_official_prompt: bool = True) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.math_scenario.MATHScenario", args={"subject": subject, "level": level}
    )

    instructions = OFFICIAL_MATH_INSTRUCTIONS
    if use_official_prompt:
        instructions = OFFICIAL_MATH_PROMPT

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=instructions,
        max_train_instances=0 if use_official_prompt else 8,  # TODO: @Frieda @Tony W. - Justify/explain
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.0,
        stop_sequences=["$", "###", "\n"],  # TODO: @Frieda @Tony W. - Justify/explain
        max_tokens=20,
        input_prefix="\nProblem: ",
        output_prefix="\nAnswer: $",
        instance_prefix="$\n###",
    )

    return RunSpec(
        name=f"math:subject={subject},level={level}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_math_metrics(),
    )


def get_boolq_spec(only_contrast=False) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.boolq_scenario.BoolQScenario", args={"only_contrast": only_contrast})

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
        max_tokens=1,
        temperature=0.0,
    )
    return RunSpec(
        name="boolq" + (":only_contrast=True" if only_contrast else ""),
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_lsat_qa_spec(task: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.lsat_qa_scenario.LSATScenario", args={"task": task})

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        instructions="The following are multiple choice questions (with answers).",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        max_train_instances=2,  # TODO: @Dor - Justify
        model="openai/davinci",
        max_eval_instances=None,
        num_outputs=1,
        # TODO: @Dor - please add temperature, max_tokens, and stop_sequences
    )

    return RunSpec(
        name=f"lsat_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_imdb_spec(only_contrast=False) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.imdb_scenario.IMDBScenario", args={"only_contrast": only_contrast})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="Passage: ",
        output_prefix="\nSentiment: ",
        num_train_trials=1,
        max_train_instances=5,
        model="openai/davinci",
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,  # full dataset has 25k test inputs
        num_outputs=1,
        max_tokens=5,  # should be one or two tokens but just in case
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name="imdb" + (":only_contrast=True" if only_contrast else ""),
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_babi_qa_spec(task: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.babi_qa_scenario.BabiQAScenario", args={"task": task})

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
        max_tokens=2 if task == "19" else 1,
        # TODO: @Dor - I think we should change this since it assumes tokenization behavior.
        # max_tokens=5,
        temperature=0.0,
        stop_sequences=["\n"],
    )
    return RunSpec(
        name=f"babi_qa:task={task}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_copyright_spec(datatag="pilot", **unused_kwargs) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.copyright_scenario.CopyrightScenario", args=dict(datatag=datatag))

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        num_train_trials=1,
        temperature=0.5,
        max_eval_instances=None,  # TODO: Modify this as necessary.
        num_outputs=1,  # TODO: More is better; but we're subject to compute constraints.
        model="openai/davinci",
        max_tokens=1000,  # TODO: Can be anything up to context window size; but we're subject to compute constraints.
    )

    return RunSpec(
        name=f"copyright:datatag={datatag}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_copyright_metrics({"normalize_by_prefix_length": True}),
    )


def get_disinformation_spec(capability: str = "reiteration") -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.disinformation_scenario.DisinformationScenario", args={"capability": capability}
    )

    if capability == "reiteration":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="Write headlines that support the thesis.",
            input_prefix="",
            output_prefix="",
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
        metrics = get_disinformation_metrics()
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
        metrics = get_toxicity_metrics()
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

    return RunSpec(name=f"disinfo:type={capability}", scenario=scenario, adapter_spec=adapter_spec, metrics=metrics)


def get_code_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.code_scenario.CodeScenario", args={"dataset": dataset})

    if dataset == "HumanEval":
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            instructions="",
            max_train_instances=0,
            max_eval_instances=10000,
            num_outputs=1,
            num_train_trials=1,
            model="openai/code-davinci-001",
            temperature=0.2,
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
            max_train_instances=0,
            max_eval_instances=10000,
            num_outputs=1,
            num_train_trials=1,
            model="openai/code-davinci-001",
            temperature=0.2,
            stop_sequences=["'''", "---", '"""', "\n\n\n"],
            max_tokens=600,
            input_prefix="",
            output_prefix="",
        )

    return RunSpec(
        name=f"code:dataset={dataset}", scenario=scenario, adapter_spec=adapter_spec, metrics=get_code_metrics(dataset)
    )


def get_natural_qa_spec(mode: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.natural_qa_scenario.NaturalQAScenario", args={"mode": mode})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
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
    )


def get_the_pile_spec(subset: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.the_pile_scenario.ThePileScenario", args={"subset": subset})

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
    )


def get_ice_spec(**kwargs) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.ice_scenario.ICEScenario", args=kwargs)

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
    )


def get_narrativeqa_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.narrativeqa_scenario.NarrativeQAScenario", args=dict())

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
    )


def get_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.synthetic_reasoning_scenario.SyntheticReasoningScenario", args={"mode": mode},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,  # TODO: @Tony W. - Justify
        max_eval_instances=None,
        num_outputs=3,  # TODO: @Tony W. - Justify
        num_train_trials=1,
        model="openai/davinci",
        temperature=1.0,  # TODO: @Tony W. - Justify; should it be 0.0
        stop_sequences=["\n"],
        max_tokens=20,  # TODO: @Tony W. - Justify
        input_prefix="",
        output_prefix="| Target: ",
    )
    return RunSpec(
        name=f"synthetic_reasoning:mode={mode}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_wikitext_103_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.wikitext_103_scenario.Wikitext103Scenario", args=dict())

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
        name="wikitext_103", scenario=scenario, adapter_spec=adapter_spec, metrics=get_basic_metrics({"names": []}),
    )


def get_blimp_spec(phenomenon: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.blimp_scenario.BLiMPScenario", args={"phenomenon": phenomenon})

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING_MINIMAL_PAIRS,
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
        name=f"blimp:phenomenon={phenomenon}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
    )


def get_xsum_summarization_spec(temperature: float = 0.3) -> RunSpec:
    # TODO: @Faisal remove this parameter once testing is done.
    scenario = ScenarioSpec(
        class_name="benchmark.summarization_scenario.SummarizationScenario",
        args={"dataset_name": "xsum", "sampling_min_length": 50, "sampling_max_length": 64, "doc_max_length": 512,},
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
        max_tokens=60,  # From Lewis et al. 2019 (https://arxiv.org/pdf/1910.13461.pdf)
        temperature=temperature,  # Temporary change for testing.
        stop_sequences=["}"],  # Summary is enclosed in curly braces. Worked more reliably than newline.
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["rouge-1", "rouge-2", "rouge-l"]}),  # TODO: Add faithfulness metrics later
    )


def get_cnndm_summarization_spec(temperature: float = 0.3) -> RunSpec:
    # TODO: @Faisal remove this parameter once testing is done.
    scenario = ScenarioSpec(
        class_name="benchmark.summarization_scenario.SummarizationScenario",
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
        metrics=get_basic_metrics({"names": ["rouge-1", "rouge-2", "rouge-l"]}),  # TODO: Add faithfulness metrics later
    )


def get_empatheticdialogues_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.dialogue_scenarios.EmpatheticDialoguesScenario", args={})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="\nBEGIN DIALOGUE\n",
        num_train_trials=1,
        max_train_instances=5,
        model="ai21/j1-large",
        max_eval_instances=100,  # TODO: @Amelia @Ashwin @Ines - Justify
        num_outputs=1,
        max_tokens=50,  # TODO: @Amelia @Ashwin @Ines - Justify
        temperature=0.9,  # TODO: @Amelia @Ashwin @Ines - Justify
        # TODO: @Amelia @Ashwin @Ines - Add stop sequences
    )

    return RunSpec(
        name="empatheticdialogues",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match", "quasi_exact_match"]}),
    )


def get_dyck_language_spec(num_parenthesis_pairs: int) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.dyck_language_scenario.DyckLanguageScenario",
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
        max_train_instances=3,  # TODO: @Mirac - Justify
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        stop_sequences=["\n"],
        max_tokens=5,  # TODO: @Mirac - Justify
        num_outputs=1,
    )

    return RunSpec(
        name=f"dyck_language_np={int(num_parenthesis_pairs)}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match_indicator"]}),
    )


def get_legal_support_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.legal_support_scenario.LegalSupportScenario", args={},)

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        instructions="Which statement best supports the passage?",
        input_prefix="Passage: ",
        output_prefix="\nAnswer: ",
        model="openai/davinci",
        temperature=0.0,
        max_train_instances=3,  # TODO: @Neel - Justify
        max_eval_instances=SIMPLE_METRIC_MAX_EVAL_INSTANCES,
        num_outputs=10,  # TODO: @Neel - Justify
        stop_sequences=["\n"],
    )

    return RunSpec(
        name="legal_support",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_entity_matching_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.entity_matching_scenario.EntityMatchingScenario", args={"dataset": dataset}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Are Row A and Row B the same? Yes or No?",
        input_prefix="",
        output_prefix="\n ",
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
        name=f"entity_matching:dataset={dataset}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_entity_data_imputation_spec(dataset: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.entity_data_imputation_scenario.EntityDataImputationScenario", args={"dataset": dataset}
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Generate the missing value in the row.",
        input_prefix="",
        output_prefix="\n ",
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
    "commonsense_qa": get_commonsense_qa_spec,
    "lsat_qa": get_lsat_qa_spec,
    "quac": get_quac_spec,
    "wikifact": get_wikifact_spec,
    "babi_qa": get_babi_qa_spec,
    "real_toxicity_prompts": get_real_toxicity_prompts_spec,
    "summarization_xsum": get_xsum_summarization_spec,
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
