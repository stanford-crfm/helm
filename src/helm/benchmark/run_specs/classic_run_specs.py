from typing import Any, Dict, List, Optional, Set

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_RANKING_BINARY,
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.binary_ranking_adapter import BinaryRankingAdapter
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_generation_adapter_spec,
    get_language_modeling_adapter_spec,
    get_machine_translation_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_ranking_binary_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_bias_metric_specs,
    get_classification_metric_specs,
    get_copyright_metric_specs,
    get_disinformation_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_language_modeling_metric_specs,
    get_numeracy_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path
from helm.common.hierarchical_logger import hlog, htrack


############################################################
# Run specs


@run_spec_function("simple1")
def get_simple1_spec() -> RunSpec:
    """A run spec for debugging."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.simple_scenarios.Simple1Scenario",
        args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 10, "num_test_instances": 10},
    )
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.\n",
        max_train_instances=5,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=3,
        model="simple/model1",
        model_deployment="simple/model1",
        temperature=1,
        stop_sequences=["."],
    )
    return RunSpec(
        name="simple1",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
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
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.bbq_metrics.BBQMetric", args={})
    ] + get_exact_match_metric_specs()

    return RunSpec(
        name=f"bbq:subject={subject},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["bbq"],
    )


# Ranking


@run_spec_function("msmarco")
def get_msmarco_spec(track: str, valid_topk: Optional[int] = None) -> RunSpec:
    from helm.benchmark.scenarios.msmarco_scenario import MSMARCOScenario

    valid_topk = None if valid_topk is None else int(valid_topk)
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.msmarco_scenario.MSMARCOScenario",
        args={"track": track, "valid_topk": valid_topk},
    )

    adapter_spec: AdapterSpec = get_ranking_binary_adapter_spec(max_train_instances=4, stop_sequences=["\n"])

    # Names of the measures we want to compute.
    measure_names = MSMARCOScenario.MEASURE_NAMES[track]
    multiple_relevance_values = set(MSMARCOScenario.GOLD_RELATIONS[track]) != {1}

    metric_specs: List[MetricSpec] = [
        MetricSpec(
            class_name="helm.benchmark.metrics.ranking_metrics.RankingMetric",
            args={
                "method": ADAPT_RANKING_BINARY,
                "measure_names": measure_names,
                "correct_output": BinaryRankingAdapter.RANKING_CORRECT_LABEL,
                "wrong_output": BinaryRankingAdapter.RANKING_WRONG_LABEL,
                "rank": valid_topk,
                "multiple_relevance_values": multiple_relevance_values,
            },
        ),
    ] + get_basic_metric_specs(names=[])

    return RunSpec(
        name=f"msmarco:track={track},valid_topk={valid_topk}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
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
    from helm.benchmark.scenarios.commonsense_scenario import (
        CommonSenseQAScenario,
        HellaSwagScenario,
        OpenBookQA,
        PiqaScenario,
        SiqaScenario,
    )

    # TODO Split these into their own run_spec_function.
    if dataset == HellaSwagScenario.name:
        scenario_spec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.commonsense_scenario.HellaSwagScenario", args={}
        )
    elif dataset == OpenBookQA.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.OpenBookQA", args={})
    elif dataset == CommonSenseQAScenario.name:
        scenario_spec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.commonsense_scenario.CommonSenseQAScenario", args={}
        )
    elif dataset == SiqaScenario.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.SiqaScenario", args={})
    elif dataset == PiqaScenario.name:
        scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.commonsense_scenario.PiqaScenario", args={})
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
        metric_specs=get_language_modeling_metric_specs([]),
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
        metric_specs=get_basic_metric_specs(["f1_set_match", "iou_set_match", "exact_set_match"])
        + get_generative_harms_metric_specs(),
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
        metric_specs=get_basic_metric_specs(["exact_match_indicator", "final_number_exact_match"])
        + get_generative_harms_metric_specs(),
        groups=["gsm"],
    )


@run_spec_function("raft")
def get_raft_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.raft_scenario import RAFTScenario, get_raft_instructions

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.raft_scenario.RAFTScenario", args={"subset": subset}
    )

    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), RAFTScenario.name)
    adapter_spec = get_generation_adapter_spec(
        instructions=get_raft_instructions(subset, scenario_cache_path),
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
    from ..scenarios.numeracy_scenario import RELTYPE_INFO, get_numeracy_adapter_spec

    run_solver_bool: bool = True if run_solver == "True" else False
    del run_solver
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
        metric_specs=get_numeracy_metric_specs(run_solver_bool),
        groups=["numeracy"],
    )


@run_spec_function("math")
def get_math_spec(
    subject: str,
    level: str,
    use_official_examples: str = "False",
    use_chain_of_thought: str = "False",
) -> RunSpec:
    # Convert to bools and remove the str versions
    use_official_examples_bool: bool = use_official_examples == "True"
    use_chain_of_thought_bool: bool = use_chain_of_thought == "True"
    del use_official_examples
    del use_chain_of_thought

    if use_chain_of_thought_bool:
        assert not use_official_examples_bool, "Cannot use official examples when use_chain_of_thought is True."
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.math_scenario.MATHScenario",
        args={
            "subject": subject,
            "level": level,
            "use_official_examples": use_official_examples_bool,
            "use_chain_of_thought": use_chain_of_thought_bool,
        },
    )

    if use_chain_of_thought_bool:  # Include the solution in the output as per https://arxiv.org/abs/2201.11903
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
        f"use_official_examples={use_official_examples_bool},use_chain_of_thought={use_chain_of_thought_bool}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(
            ["math_equiv_chain_of_thought" if use_chain_of_thought_bool else "math_equiv"]
        )
        + get_generative_harms_metric_specs(),
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
    from helm.benchmark.scenarios.copyright_scenario import datatag2hash_code

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
        metric_specs=get_language_modeling_metric_specs([]),
        groups=["the_pile"],
    )


@run_spec_function("ice")
def get_ice_spec(**kwargs) -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.ice_scenario.ICEScenario", args=kwargs)

    return RunSpec(
        name="ice" + (":" if len(kwargs) > 0 else "") + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())),
        scenario_spec=scenario_spec,
        adapter_spec=get_language_modeling_adapter_spec(),
        metric_specs=get_language_modeling_metric_specs([]),
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
        metric_specs=get_language_modeling_metric_specs([]),
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
    from helm.benchmark.scenarios.big_bench_scenario import BIGBenchScenario

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
    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), BIGBenchScenario.name)
    big_bench_task: Dict = BIGBenchScenario.download_and_get_task(scenario_cache_path, task, subtask)

    # The JSON schema for BIG-bench can be found here:
    # https://github.com/google/BIG-bench/blob/main/docs/doc.md#json-schema.
    # "metrics" is a required field. The default values were populated using the link above.
    adapter_spec = AdapterSpec(
        method=get_adaptation_method(big_bench_task["metrics"]),
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
        metric_specs=get_metric_specs(big_bench_task["metrics"]),
        groups=[f"big_bench_{task}"],
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
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["wmt_14"],
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
        metric_specs=get_basic_metric_specs(["exact_match", "quasi_exact_match"]),
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


@run_spec_function("lm_entry")
def get_lm_entry_spec(task: str, method: str = ADAPT_GENERATION) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lm_entry_scenario.LMEntryScenario",
        args={"task": task},
    )
    adapter_spec: AdapterSpec
    metric_specs: List[MetricSpec]

    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        if task in ["first_letter", "last_letter", "first_word", "last_word", "word_before", "word_after"]:
            raise ValueError(f"Task {task} cannot be cast to multiple choice.")

        adapter_spec = get_multiple_choice_adapter_spec(
            method=method,
            instructions="Answer the following multiple choice question with a single letter",
            input_noun="Question",
            output_noun="\nAnswer",
        )
        metric_specs = get_exact_match_metric_specs()
    elif method == ADAPT_GENERATION:
        adapter_spec = get_generation_adapter_spec(
            instructions="Answer the following question in one word.",
            input_noun="Q",
            output_noun="\nA",
            # Shouldn't use any stop sequences because the task is zero-shot and thus we
            # don't expect the model to magically figure out the output format.
            stop_sequences=[],
            # Set max_tokens to save tokens. The answer is a word so 10 tokens should suffice.
            max_tokens=10,
        )
        # It makes no sense to include non-quasi exact match metrics for this task.
        metric_specs = get_basic_metric_specs(["quasi_exact_match", "quasi_prefix_exact_match", "f1_score"])
    else:
        raise ValueError(f"Unknown method: {method}")

    return RunSpec(
        name=f"lm_entry:task={task},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lm_entry"],
    )
