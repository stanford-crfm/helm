from typing import List, Optional, Dict

from .adaptation.adapter_spec import AdapterSpec
from .adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL, ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL
from .metrics.metric import MetricSpec
from .run_specs import run_spec_function, get_exact_match_metric_specs, get_open_ended_generation_metric_specs
from .runner import RunSpec
from .scenarios.scenario import ScenarioSpec


############################################################
# Prototypical adapter specs for VLM evaluation


def get_generation_adapter_spec(
    instructions: str = "",
    input_prefix: str = "",
    input_suffix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_tokens: int = 100,
    stop_sequences: Optional[List[str]] = None,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        global_prefix="",
        instructions=instructions,
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        instance_prefix="\n",
        # We focus on zero-shot evaluation for now as most open VLMs only support a single image input
        max_train_instances=0,
        num_outputs=1,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences if stop_sequences is not None else [],
        random=None,
    )


def get_short_answer_generation_adapter_spec():
    return get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=20,
    )


def get_multiple_choice_joint_adapter_spec(
    input_noun: Optional[str],
    output_noun: str,
    max_train_instances: int = 0,
    num_outputs: int = 1,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
        global_prefix="",
        instructions="Answer the multiple choice question by just giving the letter of the correct answer.",
        input_prefix=f"{input_noun}: " if input_noun is not None else "",
        input_suffix="\n",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        instance_prefix="\n",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=1,
        stop_sequences=["\n"],
        temperature=0.0,
        random=None,
    )


############################################################
# VHELM metric specs


def get_image2structure_metric_specs(
    metric_names: Optional[List[str]] = None, args: Optional[Dict] = None, normalize_by_white_score: bool = False
) -> List[MetricSpec]:
    from .metrics.vision_language.image_metrics import ImageMetric

    if metric_names is None:
        metric_names = [
            ImageMetric.EARTH_MOVER_SIMILARITY,
            ImageMetric.PIXEL_SIMILARITY,
            ImageMetric.SIFT_SIMILARITY,
            ImageMetric.LPIPS_SIMILARITY,
        ]
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.vision_language.image2structure.latex_metrics.LatexMetric",
            args={"metric_names": metric_names, "normalize_by_white_score": normalize_by_white_score, **args},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "edit_similarity"},
        ),
    ]


############################################################
# VHELM run specs


@run_spec_function("chart2csv")
def get_chart2csv_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.chart2csv_scenario.Chart2CSVScenario",
        args={},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Generate the CSV from the chart.",
        max_tokens=500,
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "chart2csv"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("hateful_memes")
def get_hateful_memes_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.hateful_memes_scenario.HatefulMemesScenario", args={}
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Answer Yes or No without an explanation.",
        max_tokens=3,
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "hateful_memes"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("viz_wiz")
def get_viz_wiz_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.viz_wiz_scenario.VizWizScenario", args={}
    )
    adapter_spec: AdapterSpec = get_short_answer_generation_adapter_spec()
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "viz_wiz"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("vqa")
def get_vqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.vqa_scenario.VQAScenario", args={}
    )
    adapter_spec: AdapterSpec = get_short_answer_generation_adapter_spec()
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_open_ended_generation_metric_specs()

    run_spec_name: str = "vqa"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("image2latex")
def get_image2latex_latex_spec(subject: str, recompile_prompt: bool = True, args: Optional[Dict] = None) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.latex_scenario.LatexScenario",
        args={"subject": subject, "recompile_prompt": recompile_prompt},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = get_image2structure_metric_specs(args=args, normalize_by_white_score=False)

    run_spec_name: str = "image2latex"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mmmu")
def get_mmmu_spec(subject: str, question_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mmmu_scenario.MMMUScenario",
        args={"subject": subject, "question_type": question_type},
    )

    adapter_spec: AdapterSpec
    if question_type == "open":
        adapter_spec = get_short_answer_generation_adapter_spec()
    elif question_type == "multiple-choice":
        adapter_spec = get_multiple_choice_joint_adapter_spec(
            input_noun=None, output_noun="Answer", max_train_instances=0
        )
    else:
        raise ValueError(f"Invalid question type: {question_type}")

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "mmmu"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject},question_type={question_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("heim_human_eval")
def get_heim_human_eval_spec(question_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.heim_human_eval_scenario.HEIMHumanEvalScenario",
        args={"question_type": question_type},
    )
    adapter_spec: AdapterSpec = get_multiple_choice_joint_adapter_spec(
        input_noun=None,
        output_noun="Answer",
        num_outputs=1,
        max_train_instances=0,
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "heim_human_eval"
    return RunSpec(
        name=f"{run_spec_name}:question_type={question_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
