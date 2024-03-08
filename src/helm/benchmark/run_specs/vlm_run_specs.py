"""Run spec functions for evaluating Vision-Language Models."""

from typing import List, Optional, Dict

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec


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
    generation_type: str,
    metric_names: Optional[List[str]] = None,
    args: Optional[Dict] = None,
    include_edit_similarity: bool = True,
) -> List[MetricSpec]:
    from helm.benchmark.metrics.vision_language.image_metrics import AnnotatedImageMetrics

    if metric_names is None:
        metric_names = [
            AnnotatedImageMetrics.PIXEL_SIMILARITY,
            AnnotatedImageMetrics.FID_SIMILARITY,
            AnnotatedImageMetrics.EDIT_SIMILARITY,
        ]
    if include_edit_similarity:
        metric_names.append(AnnotatedImageMetrics.EDIT_SIMILARITY)
    if args is None:
        args = {}
    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.vision_language.image_metrics.AnnotatedImageMetrics",
            args={
                "generation_type": generation_type,
                "metric_names": metric_names,
                **args,
            },
        ),
    ]
    return metric_specs


############################################################
# VHELM run specs


@run_spec_function("a_okvqa")
def get_a_okvqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.a_okvqa_scenario.AOKVQAScenario",
        args={},
    )

    adapter_spec: AdapterSpec = get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "a_okvqa"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("chart2csv")
def get_chart2csv_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.chart2csv_scenario.Chart2CSVScenario",
        args={},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Generate the CSV for the chart. Some of the labels may be missing due to the size of the chart. "
        "Please infer the missing labels based on the surrounding context. "
        "Just give the CSV without any explanation.",
        max_tokens=1000,
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


@run_spec_function("crossmodal_3600")
def get_crossmodal_3600_spec(location: str, language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.crossmodal_3600_scenario.Crossmodal3600Scenario",
        args={"location": location, "language": language},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(max_tokens=20)
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_open_ended_generation_metric_specs()

    run_spec_name: str = "crossmodal_3600"
    return RunSpec(
        name=f"{run_spec_name}:location={location},language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("flickr30k")
def get_flickr30k_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.flickr30k_scenario.Flickr30KScenario", args={}
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Generate a short caption for the following image", max_tokens=20
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_open_ended_generation_metric_specs()

    run_spec_name: str = "flickr30k"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("gqa")
def get_gqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.gqa_scenario.GQAScenario", args={}
    )
    adapter_spec: AdapterSpec = get_short_answer_generation_adapter_spec()
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_open_ended_generation_metric_specs()

    run_spec_name: str = "gqa"
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
    adapter_spec = get_multiple_choice_joint_adapter_spec(input_noun=None, output_noun="Answer", max_train_instances=0)
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
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_open_ended_generation_metric_specs()

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
def get_image2latex_spec(subset: str, recompile_prompt: bool = False, args: Optional[Dict] = None) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.latex_scenario.LatexScenario",
        args={"subset": subset, "recompile_prompt": recompile_prompt},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = get_image2structure_metric_specs(
        generation_type="latex",
        args=args,
        include_edit_similarity=True,
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.latex_compiler_annotator.LatexCompilerAnnotator",
        )
    ]

    run_spec_name: str = "image2latex"
    return RunSpec(
        name=f"{run_spec_name}:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
        annotators=annotator_specs,
    )


@run_spec_function("image2webpage")
def get_image2webpage_spec(subset: str, recompile_prompt: bool = False, args: Optional[Dict] = None) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.webpage_scenario.WebpageScenario",
        args={"subset": subset, "recompile_prompt": recompile_prompt},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = get_image2structure_metric_specs(
        generation_type="webpage",
        args=args,
        include_edit_similarity=True,
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.webpage_compiler_annotator.WebpageCompilerAnnotator",
        )
    ]

    run_spec_name: str = "image2webpage"
    return RunSpec(
        name=f"{run_spec_name}:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
        annotators=annotator_specs,
    )


@run_spec_function("math_vista")
def get_math_vista_spec(grade: str, question_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.math_vista_scenario.MathVistaScenario",
        args={"grade": grade, "question_type": question_type},
    )

    adapter_spec: AdapterSpec
    if question_type == "free_form":
        adapter_spec = get_short_answer_generation_adapter_spec()
    elif question_type == "multi_choice":
        adapter_spec = get_multiple_choice_joint_adapter_spec(
            input_noun=None, output_noun="Answer", max_train_instances=0
        )
    else:
        raise ValueError(f"Invalid question type: {question_type}")

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "math_vista"
    return RunSpec(
        name=f"{run_spec_name}:grade={grade},question_type={question_type}",
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


@run_spec_function("sheetmusic2lilypond")
def get_sheetmusic2lilypond_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.sheetmusic2lilypond_scenario."
        "SheetMusic2LilyPondScenario",
        args={},
    )
    adapter_spec: AdapterSpec = get_generation_adapter_spec(
        instructions="Generate the LilyPond code for the following sheet music. "
        "Just give the LilyPond code without any explanation.",
        max_tokens=1500,
    )

    metric_specs: List[MetricSpec] = get_image2structure_metric_specs(
        generation_type="lilypond",
        include_edit_similarity=False,
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.lilypond_compiler_annotator.LilyPondAnnotator",
        )
    ]

    run_spec_name: str = "sheetmusic2lilypond"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        annotators=annotator_specs,
        groups=[run_spec_name],
    )
