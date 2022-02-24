from typing import List, Dict

from proxy.openai_client import OPENAI_END_OF_TEXT_TOKEN
from .scenario import ScenarioSpec
from .adapter import (
    AdapterSpec,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE,
    ADAPT_GENERATION,
    ADAPT_LANGUAGE_GENERATION,
)
from .metric import MetricSpec
from .runner import RunSpec


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


def get_toxicity_metrics(group_tags: List[str]) -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={"group_tags": group_tags})]


def get_lpm_metrics() -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.lpm_metrics.LPMMetric", args={})]


############################################################


def get_run_spec1() -> RunSpec:
    """An run spec for debugging."""
    return RunSpec(
        name="run1",
        scenario=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metrics=get_basic_metrics({"names": []}),
    )


def get_mmlu_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.mmlu_scenario.MMLUScenario", args={"subject": subject},)

    def format(subject: str):
        return subject.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        conditioning_prefix="",
        instructions=f"The following are multiple choice questions (with answers) about {format(subject)}.",
        input_prefix="",
        output_prefix="\nAnswer: ",
        max_train_instances=5,
        max_eval_instances=1000,
        num_outputs=10,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0,
    )

    return RunSpec(
        name=f"mmlu_subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_twitter_aae_spec(demographic: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.twitter_aae_scenario.TwitterAAEScenario", args={"demographic": demographic},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        conditioning_prefix=OPENAI_END_OF_TEXT_TOKEN,
        instructions="",
        input_prefix="",
        output_prefix="",
        max_train_instances=0,
        max_eval_instances=1000,  # TODO: remove this line once deployed, so we can cache everything in prod
        num_outputs=1,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0,
        max_tokens=0,
    )

    return RunSpec(
        name="twitter_aae_demographic={demographic}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": []}),
    )


def get_real_toxicity_prompts_spec() -> RunSpec:
    from .real_toxicity_prompts_scenario import TOXIC_TAG, NONTOXIC_TAG

    scenario = ScenarioSpec(class_name="benchmark.real_toxicity_prompts_scenario.RealToxicityPromptsScenario", args={})
    # Create AdapterSpec based on the RealToxicityPrompts paper: https://arxiv.org/pdf/2009.11462.pdf
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=0,
        max_eval_instances=1000,  # TODO: set this to 100,000 once deployed, so we can cache everything in prod
        model="openai/davinci",  # "...we use the 175B parameter GPT-3 model, also known as DA VINCI in the OpenAI API"
        temperature=1,  # "We use a temperature of 1 during generation..."
        max_tokens=20,  # "We generate up to 20 tokens per example..."
        num_outputs=25,  # "...the expected maximum toxicity over k = 25 generations..."
    )
    return RunSpec(
        name="real_toxicity_prompts",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_toxicity_metrics([TOXIC_TAG, NONTOXIC_TAG]),
    )


def get_lpm_spec(difficulty: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.lpm_scenario.LPMScenario", args={"difficulty": difficulty},)

    def format(difficulty: str):
        return difficulty.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0.2,
        stop_sequences=["\n"],
        max_tokens=20,
        input_prefix="Rules:\n",
        output_prefix="",
    )

    return RunSpec(
        name=f"lpm_difficulty={difficulty}", scenario=scenario, adapter_spec=adapter_spec, metrics=get_lpm_metrics()
    )


def get_synthetic_reasoning_spec(mode: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.synthetic_reasoning_scenario.SyntheticReasoningScenario", args={"mode": mode},
    )

    def format(difficulty: str):
        return difficulty.replace("_", " ")

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,
        max_eval_instances=10,
        num_outputs=3,
        num_train_trials=1,
        model="openai/davinci",
        # model="simple/model1",
        temperature=0.2,
        stop_sequences=["\n"],
        max_tokens=20,
        input_prefix="",
        output_prefix="| Target: ",
    )

    return RunSpec(
        name=f"synrea_mode={mode}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )



def get_dyck_language_spec(num_parenthesis_pairs: int) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.dyck_language_scenario.DyckLanguageScenario", 
        args={"num_parenthesis_pairs": int(num_parenthesis_pairs)},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_GENERATION,
        conditioning_prefix="",
        instructions=f"Please complete the rest of the following Dyck sequence, making sure that the parentheses are closed properly. ",
        input_prefix="Input: ",
        output_prefix="",
        model="openai/davinci",
        temperature=0.0,
        max_train_instances=3,
        max_eval_instances=100,
        stop_sequences=["\n"],
        max_tokens=10,
        num_outputs=1,
    )

    return RunSpec(
        name=f"dyck_language_np={int(num_parenthesis_pairs)}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["iou_set_match", "exact_match"]}),
    )
