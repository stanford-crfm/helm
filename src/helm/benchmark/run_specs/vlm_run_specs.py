"""Run spec functions for evaluating Vision-Language Models."""

from typing import List, Optional, Dict

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
)
from helm.benchmark.scenarios.vision_language.image2structure.image2structure_scenario import DIFFICULTY_ALL
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
    get_generative_harms_metric_specs,
    get_basic_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec


############################################################
# Prototypical adapter specs for VLM evaluation


def _get_generation_adapter_spec(
    instructions: str = "",
    input_prefix: str = "",
    input_suffix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_tokens: int = 100,
    max_train_instances: int = 0,
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
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences if stop_sequences is not None else [],
        temperature=0.0,
        random=None,
    )


def _get_short_answer_generation_adapter_spec(instructions: Optional[str] = None) -> AdapterSpec:
    return _get_generation_adapter_spec(
        instructions=(
            "Just give a short answer without answering in a complete sentence."
            if instructions is None
            else instructions
        ),
        max_tokens=20,
    )


def _get_captioning_adapter_spec() -> AdapterSpec:
    return _get_generation_adapter_spec(
        instructions="Generate a caption for the following image. The caption should be short and does "
        "not need to be a complete sentence.",
        max_tokens=20,
    )


def get_open_end_answer_generation_adapter_spec():
    return _get_generation_adapter_spec(
        instructions="Follow the given instruction and give your complete answer.",
        max_tokens=100,
    )


def _get_multiple_choice_joint_adapter_spec(
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


def _get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4", "cider"]
    )


def _get_image2structure_metric_specs(
    generation_type: str,
    metric_names: Optional[List[str]] = None,
    args: Optional[Dict] = None,
    include_edit_similarity: bool = True,
    size_handling_method: str = "resize",
) -> List[MetricSpec]:
    from helm.benchmark.metrics.vision_language.image_metrics import AnnotatedImageMetrics

    if metric_names is None:
        metric_names = [
            AnnotatedImageMetrics.PIXEL_SIMILARITY,
            AnnotatedImageMetrics.FID_SIMILARITY,
            AnnotatedImageMetrics.BLOCK_EMD,
            AnnotatedImageMetrics.EARTH_MOVER_SIMILARITY,
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
                "size_handling_method": size_handling_method,
                **args,
            },
        ),
    ]
    return metric_specs + get_basic_metric_specs([])


def _get_prometheus_vision_critique_metric_specs(num_respondents: int, max_tokens: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.prometheus_vision_critique_metrics.PrometheusVisionCritiqueMetric",
            args={
                "num_respondents": num_respondents,
                "max_tokens": max_tokens,
            },
        )
    ]


def _get_gpt4v_critique_originality_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.gpt4v_originality_critique_metrics.GPT4VCritiqueMetric",
            args={
                "num_respondents": num_respondents,
            },
        )
    ]


def _get_vibe_eval_critique_metric_specs(num_respondents: int, max_tokens: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.reka_vibe_critique_metrics.RekaVibeCritiqueMetric",
            args={
                "num_respondents": num_respondents,
                "max_tokens": max_tokens,
            },
        )
    ]


############################################################
# VHELM run specs


@run_spec_function("a_okvqa")
def get_a_okvqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.a_okvqa_scenario.AOKVQAScenario",
        args={},
    )

    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
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
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
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
def get_crossmodal_3600_spec(location: str, language: str, num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.crossmodal_3600_scenario.Crossmodal3600Scenario",
        args={"location": location, "language": language},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Answer the question with a complete sentence in plain words",
        max_tokens=20,
    )

    metric_specs: List[MetricSpec] = (
        _get_prometheus_vision_critique_metric_specs(
            num_respondents=num_respondents,
            max_tokens=200,
        )
        + _get_open_ended_generation_metric_specs()
    )

    run_spec_name: str = "crossmodal_3600"
    return RunSpec(
        name=f"{run_spec_name}:location={location},language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("flickr30k")
def get_flickr30k_spec(num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.flickr30k_scenario.Flickr30KScenario", args={}
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Generate a caption for the following image in plain words. The caption should "
        "be short and needs to be a complete sentence.",
        max_tokens=30,
        max_train_instances=0,
    )
    metric_specs: List[MetricSpec] = (
        _get_prometheus_vision_critique_metric_specs(
            num_respondents=num_respondents,
            max_tokens=200,
        )
        + _get_open_ended_generation_metric_specs()
    )

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
    adapter_spec: AdapterSpec = _get_short_answer_generation_adapter_spec(
        instructions="Answer the question using a single word."
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + _get_open_ended_generation_metric_specs()

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
    adapter_spec = _get_multiple_choice_joint_adapter_spec(input_noun=None, output_noun="Answer", max_train_instances=0)
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "hateful_memes"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mm_safety_bench")
def get_mm_safety_bench_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mm_safety_bench_scenario.MMSafetyBenchScenario",
        args={"subset": subset},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(max_tokens=500)
    metric_specs: List[MetricSpec] = get_generative_harms_metric_specs(
        include_basic_metrics=True, include_generative_harms_metrics=True
    )

    run_spec_name: str = "mm_safety_bench"
    return RunSpec(
        name=f"{run_spec_name}:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mscoco_captioning")
def get_mscoco_captioning_spec(long: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mscoco_captioning_scenario.MSCOCOCaptioningScenario",
        args={},
    )

    adapter_spec: AdapterSpec
    if long:
        adapter_spec = _get_generation_adapter_spec(
            instructions="Generate a long, detailed caption for the following image.",
            max_tokens=200,
        )
    else:
        adapter_spec = _get_generation_adapter_spec(
            instructions="Generate a caption for the following image. The caption should be short and does "
            "not need to be a complete sentence.",
            max_tokens=20,
        )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + _get_open_ended_generation_metric_specs()

    run_spec_name: str = "mscoco_captioning"
    if long:
        run_spec_name += "_long"

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mscoco_categorization")
def get_mscoco_categorization_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mscoco_categorization_scenario."
        "MSCOCOCategorizationScenario",
        args={},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "mscoco_categorization"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("originality_vlm")
def get_originality_vlm_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.originality_scenario.OriginalityScenario", args={}
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(max_tokens=500)
    metric_specs: List[MetricSpec] = get_generative_harms_metric_specs(
        include_basic_metrics=True, include_generative_harms_metrics=True
    )

    run_spec_name: str = "originality_vlm"
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
    adapter_spec: AdapterSpec = _get_short_answer_generation_adapter_spec(
        # Following https://arxiv.org/abs/2310.03744
        instructions="When the provided information is insufficient, respond with 'Unanswerable'. "
        "Answer the question using a single word or phrase."
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + _get_open_ended_generation_metric_specs()

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
    # Following https://arxiv.org/abs/2310.03744
    adapter_spec: AdapterSpec = _get_short_answer_generation_adapter_spec(
        instructions='Answer the question using a single word or phrase. When the question asks "How many...", '
        "respond with just a number (e.g., 3) and not the word corresponding to the number."
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + _get_open_ended_generation_metric_specs()

    run_spec_name: str = "vqa"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("image2latex")
def get_image2latex_spec(
    subset: str, recompile_prompt: bool = False, difficulty: str = DIFFICULTY_ALL, args: Optional[Dict] = None
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.latex_scenario.LatexScenario",
        args={"subset": subset, "recompile_prompt": recompile_prompt, "difficulty": difficulty},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = _get_image2structure_metric_specs(
        generation_type="latex",
        args=args,
        include_edit_similarity=(subset != "real"),
        size_handling_method="padding",
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.latex_compiler_annotator.LatexCompilerAnnotator",
        )
    ]

    run_spec_name: str = f"image2latex:subset={subset}:difficulty={difficulty}"
    groups: List[str]
    if subset == "real":
        groups = ["image2latex_real"]
    else:
        groups = ["image2latex", f"image2latex_{difficulty}"]
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=groups,
        annotators=annotator_specs,
    )


@run_spec_function("image2webpage")
def get_image2webpage_spec(
    subset: str,
    recompile_prompt: bool = False,
    difficulty: str = DIFFICULTY_ALL,
    args: Optional[Dict] = None,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.webpage_scenario.WebpageScenario",
        args={"subset": subset, "recompile_prompt": recompile_prompt, "difficulty": difficulty},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = _get_image2structure_metric_specs(
        generation_type="webpage",
        args=args,
        include_edit_similarity=(subset != "real"),
        size_handling_method="none",
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.webpage_compiler_annotator.WebpageCompilerAnnotator",
        )
    ]

    run_spec_name: str = f"image2webpage:subset={subset}:difficulty={difficulty}"
    groups: List[str]
    if subset == "real":
        groups = ["image2webpage_real"]
    else:
        groups = ["image2webpage", f"image2webpage_{difficulty}"]
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=groups,
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
        adapter_spec = _get_short_answer_generation_adapter_spec(
            instructions="Just give the numerical answer without showing the steps, the unit, or percentage symbol."
        )
    elif question_type == "multi_choice":
        adapter_spec = _get_multiple_choice_joint_adapter_spec(
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


@run_spec_function("image2musicsheet")
def get_image2musicsheet_spec(difficulty: str = DIFFICULTY_ALL, args: Optional[Dict] = None) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.image2structure.musicsheet_scenario.MusicSheetScenario",
        # There os only one subset for music sheets
        args={"subset": "music", "recompile_prompt": False, "difficulty": difficulty},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Just give a short answer without answering in a complete sentence.",
        max_tokens=2000,
    )
    metric_specs: List[MetricSpec] = _get_image2structure_metric_specs(
        generation_type="lilypond",
        args=args,
        include_edit_similarity=False,  # No ground truth for music sheets
        size_handling_method="padding",
    )
    annotator_specs: List[AnnotatorSpec] = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.image2structure.lilypond_compiler_annotator.LilypondCompilerAnnotator",  # noqa: E501
        )
    ]

    run_spec_name: str = f"image2musicsheet:difficulty={difficulty}"
    groups: List[str] = ["image2musicsheet", f"image2musicsheet_{difficulty}"]
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=groups,
        annotators=annotator_specs,
    )


@run_spec_function("mmmu")
def get_mmmu_spec(subject: str, question_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mmmu_scenario.MMMUScenario",
        args={"subject": subject, "question_type": question_type},
    )

    adapter_spec: AdapterSpec
    if question_type == "open":
        adapter_spec = _get_short_answer_generation_adapter_spec()
    elif question_type == "multiple-choice":
        adapter_spec = _get_multiple_choice_joint_adapter_spec(
            input_noun=None,
            output_noun="Answer",
            max_train_instances=0,
            # instructions="Refer to the figure(s) and answer the multiple choice question by responding with just "
            # "the letter of the correct answer (e.g., A, B, C, D, E).",
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


@run_spec_function("unicorn")
def get_unicorn_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.unicorn_scenario.UnicornScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Only give a yes/no or numerical answer without an explanation.",
        max_tokens=1,  # the model may generate answer with a period
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "unicorn"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("bingo")
def get_bingo_spec(subject: str, num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.bingo_scenario.BingoScenario", args={"subject": subject}
    )
    adapter_spec: AdapterSpec = _get_generation_adapter_spec(
        instructions="Answer the question with a complete and clear explanation in sentences without listing it out.",
        max_tokens=100,
        max_train_instances=0,
    )
    metric_specs: List[MetricSpec] = (
        _get_prometheus_vision_critique_metric_specs(
            num_respondents=num_respondents,
            max_tokens=200,
        )
        + _get_open_ended_generation_metric_specs()
    )

    run_spec_name: str = "bingo"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("multipanelvqa")
def get_multipanelvqa_spec(subject: str, question_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.multipanelvqa_scenario.MultipanelVQAScenario",
        args={"subject": subject, "question_type": question_type},
    )

    adapter_spec: AdapterSpec
    if question_type == "open":
        adapter_spec = _get_short_answer_generation_adapter_spec()
    elif question_type == "multiple-choice":
        adapter_spec = _get_multiple_choice_joint_adapter_spec(
            input_noun=None, output_noun="Answer", max_train_instances=0
        )
    else:
        raise ValueError(f"Invalid question type: {question_type}")

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "multipanelvqa"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject},question_type={question_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("pope")
def get_pope_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.pope_scenario.POPEScenario",
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "pope"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("seed_bench")
def get_seed_bench_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.seed_bench_scenario.SEEDBenchScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "seed_bench"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mme")
def get_mme_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mme_scenario.MMEScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "mme"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
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
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
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


@run_spec_function("pairs")
def get_pairs_spec(subset: str, person: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.pairs_scenario.PAIRSScenario",
        args={"subset": subset, "person": person},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None,
        output_noun="Answer",
        num_outputs=1,
        max_train_instances=0,
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "pairs"
    return RunSpec(
        name=f"{run_spec_name}:subset={subset},person={person}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mementos")
def get_mementos_spec(subject: str, num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.mementos_scenario.MementosScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = get_open_end_answer_generation_adapter_spec()
    metric_specs: List[MetricSpec] = (
        _get_prometheus_vision_critique_metric_specs(num_respondents=num_respondents, max_tokens=200)
        + _get_open_ended_generation_metric_specs()
    )

    run_spec_name: str = "mementos"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("vibe_eval")
def get_vibe_eval_spec(subject: str, num_respondents: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.vibe_eval_scenario.VibeEvalScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = get_open_end_answer_generation_adapter_spec()
    metric_specs: List[MetricSpec] = (
        _get_prometheus_vision_critique_metric_specs(num_respondents=num_respondents, max_tokens=200)
        + _get_open_ended_generation_metric_specs()
    )

    run_spec_name: str = "vibe_eval"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
