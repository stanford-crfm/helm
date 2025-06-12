"""Run spec functions for the HELM Capabilities leaderboard.

Website: https://crfm.stanford.edu/helm/capabilities/"""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_CHAT,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _bool_to_str(value: bool):
    return str(value).lower()


@run_spec_function("mmlu_pro")
def get_mmlu_pro_spec(subject: str, use_chain_of_thought: str = "true", use_few_shot: str = "false") -> RunSpec:
    # Convert to bools and remove the str versions
    use_chain_of_thought_bool: bool = use_chain_of_thought.lower() == "true"
    use_few_shot_bool: bool = use_few_shot.lower() == "true"
    del use_chain_of_thought
    del use_few_shot

    run_spec_name = f"mmlu_pro:subset={subject},use_chain_of_thought={_bool_to_str(use_chain_of_thought_bool)},use_few_shot={_bool_to_str(use_few_shot_bool)}"  # noqa: E501
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_pro_scenario.MMLUProScenario", args={"subject": subject}
    )
    max_train_instance_num = 5 if use_few_shot_bool else 0

    if use_chain_of_thought_bool:
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
            max_train_instances=max_train_instance_num,
            max_tokens=4096,  # original: 4000
            input_prefix="What is the correct answer to this question: ",
            input_suffix="\nChoices:\n",
            output_prefix="",
            global_suffix=(
                "Let’s think step by step. Based on your reasoning, what is the single, "
                "most likely answer choice? Format your response as follows: "
                '"The correct answer is (insert answer here)".'
            ),
        )
        return RunSpec(
            name=run_spec_name,
            scenario_spec=scenario_spec,
            adapter_spec=adapter_spec,
            metric_specs=get_basic_metric_specs([])
            + [
                MetricSpec(
                    class_name="helm.benchmark.metrics.gpqa_chain_of_thought_metric.GPQAChainOfThoughtMetric", args={}
                ),
            ],
            groups=["mmlu_pro"],
        )

    else:
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            max_train_instances=max_train_instance_num,
            max_tokens=4096,  # original: 4000
            input_prefix="What is the correct answer to this question: ",
            input_suffix="\nChoices:\n",
            output_prefix="",
            global_suffix=("Format your response as follows: " '"The correct answer is (insert answer here)".'),
        )
        return RunSpec(
            name=run_spec_name,
            scenario_spec=scenario_spec,
            adapter_spec=adapter_spec,
            metric_specs=get_exact_match_metric_specs(),
            groups=["mmlu_pro"],
        )


@run_spec_function("gpqa")
def get_gpqa_spec(subset: str, use_chain_of_thought: str = "true", use_few_shot: str = "false") -> RunSpec:
    # Convert to bools and remove the str versions
    use_chain_of_thought_bool: bool = use_chain_of_thought.lower() == "true"
    use_few_shot_bool: bool = use_few_shot.lower() == "true"
    del use_chain_of_thought
    del use_few_shot

    if not subset.startswith("gpqa_"):
        subset = "gpqa_" + subset

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.gpqa_scenario.GPQAScenario", args={"subset": subset}
    )
    max_train_instance_num = 5 if use_few_shot_bool else 0

    if use_few_shot_bool:
        if use_chain_of_thought_bool:
            adapter_spec = get_multiple_choice_adapter_spec(
                method=ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
                max_tokens=2000,  # original: 1000
                max_train_instances=max_train_instance_num,
                instructions=(
                    "Here are some example questions from experts. "
                    "An explanation is given before the final answer. "
                    "Answer the final question yourself, giving your reasoning beforehand."
                ),
                input_noun="Question",
                input_suffix="\nChoices: \n",
                reference_prefix="(A) ",
                chain_of_thought_prefix="Let's think step by step: ",
                chain_of_thought_suffix="The correct answer is ",
                output_noun="",  # will be overwritten with output_prefix
                output_prefix="",
                global_suffix=(
                    "Give step by step reasoning before you answer, and when you’re ready to answer, "
                    'please use the format "The correct answer is (insert answer here)":'
                ),
            )
        else:
            adapter_spec = get_multiple_choice_adapter_spec(
                method=ADAPT_MULTIPLE_CHOICE_JOINT,
                max_train_instances=max_train_instance_num,
                instructions=(
                    "Here are some example questions from experts. "
                    "An explanation is given before the final answer. "
                    "Answer the final question yourself, giving your reasoning beforehand."
                ),
                input_noun="Question",
                input_suffix="\nChoices: \n",
                reference_prefix="(A) ",
                output_noun="",  # will be overwritten with output_prefix
                output_prefix="The correct answer is ",
            )
    else:
        if use_chain_of_thought_bool:
            adapter_spec = AdapterSpec(
                method=ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
                max_train_instances=max_train_instance_num,
                max_tokens=4096,  # original: 1000
                input_prefix="What is the correct answer to this question: ",
                input_suffix="\nChoices:\n",
                output_prefix="",
                reference_prefix="(A) ",
                global_suffix=(
                    "Let’s think step by step. Based on your reasoning, what is the single, "
                    "most likely answer choice? Format your response as follows: "
                    '"The correct answer is (insert answer here)".'
                ),
            )
        else:
            adapter_spec = AdapterSpec(
                method=ADAPT_MULTIPLE_CHOICE_JOINT,
                max_train_instances=max_train_instance_num,
                max_tokens=4096,  # original: 1000
                input_prefix="What is the correct answer to this question: ",
                input_suffix="\nChoices:\n",
                output_prefix="",
                reference_prefix="(A) ",
                global_suffix=("Format your response as follows: " '"The correct answer is (insert answer here)".'),
            )

    metric_specs = (
        (
            get_basic_metric_specs([])
            + [
                MetricSpec(
                    class_name="helm.benchmark.metrics.gpqa_chain_of_thought_metric.GPQAChainOfThoughtMetric", args={}
                ),
            ]
        )
        if use_chain_of_thought_bool
        else get_exact_match_metric_specs()
    )

    return RunSpec(
        name=f"gpqa:subset={subset},use_chain_of_thought={_bool_to_str(use_chain_of_thought_bool)},use_few_shot={_bool_to_str(use_few_shot_bool)}",  # noqa: E501
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["gpqa"],
    )


@run_spec_function("ifeval")
def get_ifeval_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.ifeval_scenario.IFEvalScenario")

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        max_tokens=4096,  # Unknown number from paper
        num_outputs=1,
        temperature=0.0,
    )

    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.ifeval_metrics.IFEvalMetric")
    ]

    return RunSpec(
        name="ifeval",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["ifeval"],
    )


@run_spec_function("wildbench")
def get_wildbench_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.wildbench_scenario.WildBenchScenario",
        args={
            "subset": subset,
            "use_model_outputs": False,
        },
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_CHAT, input_prefix="", output_prefix="", max_tokens=2000, num_outputs=1, temperature=0.0
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.wildbench_annotator.WildBenchAnnotator")]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.wildbench_metrics.WildBenchScoreMetric")
    ]

    return RunSpec(
        name=f"wildbench:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["wildbench"],
    )


# TODO: Remove BigCodeBench from capabilities_run_specs.py because it is no longer part of HELM Capabilities
@run_spec_function("bigcodebench")
def get_bigcodebench_spec(version: str) -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bigcodebench_scenario.BigCodeBenchScenario", args={"version": version}
    )

    # Adapted from https://github.dev/bigcode-project/bigcodebench/blob/main/bigcodebench/evaluate.py
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        output_prefix="",
        max_tokens=4096,  # original: 1280
        num_outputs=1,
        temperature=0.0,
        global_prefix="Please provide a self-contained Python script "
        "that solves the following problem in a markdown code block:",
    )
    annotator_specs = [
        AnnotatorSpec(class_name="helm.benchmark.annotation.bigcodebench_annotator.BigCodeBenchAnnotator")
    ]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.bigcodebench_metrics.BigCodeBenchMetric")
    ]

    return RunSpec(
        name=f"bigcodebench:version={version}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["bigcodebench"],
    )


@run_spec_function("omni_math")
def get_omni_math_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.omni_math_scenario.OmniMATHScenario")

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Answer the question, giving your reasoning beforehand. Wrap the final answer with the \\boxed{} command.",  # noqa: E501
        input_prefix="",
        output_prefix="",
        max_tokens=4096,  # original: 2048
        num_outputs=1,
        temperature=0.0,
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.omni_math_annotator.OmniMATHAnnotator")]
    metric_specs = get_basic_metric_specs([]) + [
        MetricSpec(class_name="helm.benchmark.metrics.omni_math_metrics.OmniMATHMetric")
    ]

    return RunSpec(
        name="omni_math",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["omni_math"],
    )
