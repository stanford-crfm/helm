from typing import List, Dict, Optional, Any

from proxy.openai_client import OPENAI_END_OF_TEXT_TOKEN
from common.object_spec import ObjectSpec

from .adapter import (
    AdapterSpec,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE,
    ADAPT_GENERATION,
)
from .metric import MetricSpec
from .runner import RunSpec
from .scenario import ScenarioSpec
from .commonsense_qa_scenario import MULTI_CHOICE_QUESTION_ANSWERING_METHOD, CAUSAL_LANGUAGE_MODELING_METHOD


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

def get_bold_metrics() -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={"group_tags": group_tags})]


def get_commonsense_qa_metrics(args: Dict[str, Any]) -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.commonsense_qa_metrics.CommonSenseQAMetric", args=args)]


def get_toxicity_metrics(group_tags: List[str]) -> List[MetricSpec]:
    return [MetricSpec(class_name="benchmark.toxicity_metrics.ToxicityMetric", args={"group_tags": group_tags})]


def get_lpm_metrics() -> List[MetricSpec]:
    metric_names = {"names": ["iou_set_match", "exact_set_match"]}
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


############################################################


def construct_run_specs(spec: ObjectSpec) -> List[RunSpec]:
    """
    Takes a specification (name, args) and returns a list of `RunSpec`s.
    """
    # Note that we are abusing `spec` a bit because the name is not actually a class name.
    name = spec.class_name
    args = spec.args

    # Place these alphabetically
    if name == "bbq":
        return [get_bbq_spec()]
    if name == "bold":
        return [get_bold_spec()]
    if name == "boolq":
        return [get_boolq_spec()]
    if name == "boolq_contrast_sets":
        return [get_boolq_contrast_sets_spec()]
    if name == "copyright":
        return [get_copyright_spec(**args)]
    if name == "lpm":
        return [get_lpm_spec(**args)]
    if name == "mmlu":
        return [get_mmlu_spec(**args)]
    if name == "commonsense_qa":
        return [get_commonsense_qa_spec(**args)]
    if name == "real_toxicity_prompts":
        return [get_real_toxicity_prompts_spec()]
    if name == "simple1":
        return [get_run_spec1()]
    if name == "twitter_aae":
        return [get_twitter_aae_spec(**args)]

    raise ValueError(f"Unknown run spec: {spec}")


def get_run_spec1() -> RunSpec:
    """An run spec for debugging."""
    return RunSpec(
        name="run1",
        scenario=get_scenario_spec1(),
        adapter_spec=get_adapter_spec1(),
        metrics=get_basic_metrics({"names": []}),
    )


def get_bbq_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.bbq_scenario.BBQScenario", args={})

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE,
        conditioning_prefix="",
        instructions="The following are multiple choice questions (with answers).",
        input_prefix="",
        output_prefix="\nAnswer: ",
        max_train_instances=100,
        max_eval_instances=1000,
        num_outputs=10,
        num_train_trials=1,
        model="openai/davinci",
        temperature=0,
    )

    return RunSpec(name="bbq", scenario=scenario, adapter_spec=adapter_spec, metrics=get_bbq_metrics())

def get_bold_spec() -> RunSpec:
    from .bold_scenario import TOXIC_TAG, NONTOXIC_TAG

    scenario = ScenarioSpec(class_name="benchmark.bold_scenario.BOLDScenario", args={})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        num_train_trials=1,
        max_train_instances=0,
        max_eval_instances=1000,
        model="openai/davinci",
        temperature=1,  
        max_tokens=20, 
        num_outputs=25, 
    )
    return RunSpec(
        name="bold",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_bold_metrics,
    )



def get_mmlu_spec(subject: str) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.mmlu_scenario.MMLUScenario", args={"subject": subject})

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
        name=f"mmlu:subject={subject}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_commonsense_qa_spec(dataset: str, method: str) -> RunSpec:
    scenario = ScenarioSpec(
        class_name="benchmark.commonsense_qa_scenario.CommonSenseQAScenario",
        args={"dataset": dataset, "method": method,},
    )

    if method == MULTI_CHOICE_QUESTION_ANSWERING_METHOD:
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE,
            conditioning_prefix="",
            instructions="The following are multiple choice questions (with answers) about common sense.",
            input_prefix="",
            output_prefix="\nAnswer: ",
            max_train_instances=0,
            max_eval_instances=10,
            num_outputs=10,
            num_train_trials=1,
            model="openai/davinci",
            temperature=0,
        )
        run_spec = RunSpec(
            name=f"dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_basic_metrics({"names": ["exact_match"]}),
        )
    elif method == CAUSAL_LANGUAGE_MODELING_METHOD:
        n_choice = {"hellaswag": 4, "openbookqa": 4, "commonsenseqa": 5, "piqa": 2, "siqa": 3,}[dataset]
        adapter_spec = AdapterSpec(
            method=ADAPT_LANGUAGE_MODELING,
            conditioning_prefix="",
            instructions="",
            input_prefix="",
            output_prefix="",
            max_train_instances=0,
            max_eval_instances=10 * n_choice * 2,
            num_outputs=10,
            max_tokens=0,
            num_train_trials=1,
            model="openai/davinci",
            temperature=0,
        )
        run_spec = RunSpec(
            name=f"dataset={dataset},method={method}",
            scenario=scenario,
            adapter_spec=adapter_spec,
            metrics=get_commonsense_qa_metrics({"n_choice": n_choice}),
        )
    else:
        raise ValueError(f"Unknown commonsense QA method: {method}")

    return run_spec


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
        name=f"twitter_aae:demographic={demographic}",
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
    scenario = ScenarioSpec(class_name="benchmark.lpm_scenario.LPMScenario", args={"difficulty": difficulty})

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please solve the following problem.",
        max_train_instances=3,
        max_eval_instances=100,
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
        name=f"lpm:difficulty={difficulty}", scenario=scenario, adapter_spec=adapter_spec, metrics=get_lpm_metrics()
    )


def get_boolq_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.boolq_scenario.BoolQScenario", args={})

    # TODO: Choosing a large # of train instances results in exceeding maximum sequence length for models.
    # What's the best way to solve this?

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="\nanswer:",
        num_train_trials=1,
        max_train_instances=5,
        model="ai21/j1-large",
        max_eval_instances=50,  # TODO : Remove this once deployed
        num_outputs=1,
        max_tokens=1,
    )
    return RunSpec(
        name="boolq",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_boolq_contrast_sets_spec() -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.boolq_scenario.BoolQContrastSetScenario", args={})
    # TODO: Choosing a large # of train instances results in exceeding maximum sequence length for models.
    # What's the best way to solve this?

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="\nanswer:",
        num_train_trials=1,
        max_train_instances=5,
        model="ai21/j1-large",
        max_eval_instances=50,  # TODO : Remove this once deployed
        num_outputs=1,
        max_tokens=1,
    )
    return RunSpec(
        name="boolq_contrast_sets",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_basic_metrics({"names": ["exact_match"]}),
    )


def get_copyright_spec(pilot_study="true", **unused_kwargs) -> RunSpec:
    scenario = ScenarioSpec(class_name="benchmark.copyright_scenario.CopyrightScenario", args=dict())

    # TODO(lxuechen): Loop over models and other hyperparameter combos in the future.
    if pilot_study.lower() in ("t", "true"):
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            conditioning_prefix="",
            instructions="",
            input_prefix="",
            output_prefix="",
            max_train_instances=0,
            num_train_trials=1,
            temperature=0.7,
            # Args that are different below.
            max_eval_instances=100,
            num_outputs=1,
            model="simple/model1",
            max_tokens=60,
        )
    else:
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            conditioning_prefix="",
            instructions="",
            input_prefix="",
            output_prefix="",
            max_train_instances=0,
            num_train_trials=1,
            temperature=0.7,
            # Args that are different below.
            max_eval_instances=None,
            num_outputs=10,
            model="openai/davinci",
            max_tokens=2000,
        )

    return RunSpec(
        name=f"copyright:pilot_study={pilot_study}",
        scenario=scenario,
        adapter_spec=adapter_spec,
        metrics=get_copyright_metrics({"normalize_by_prefix_length": True}),
    )
