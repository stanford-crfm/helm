"""Run spec functions for HEIM (Holistic Evaluation of Text-To-Image Models).

Website: https://crfm.stanford.edu/heim/"""

from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.run_specs.classic_run_specs import get_basic_metric_specs
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.image_generation_parameters import ImageGenerationParameters


############################################################
# Prototypical adapter specs for text-to-image model evaluation


def get_image_generation_adapter_spec(
    num_outputs: int = 1,
    output_image_width: Optional[int] = None,
    output_image_height: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    diffusion_denoising_steps: Optional[int] = None,
    random: Optional[str] = None,
) -> AdapterSpec:
    image_generation_parameters: ImageGenerationParameters = ImageGenerationParameters(
        output_image_width=output_image_width,
        output_image_height=output_image_height,
        guidance_scale=guidance_scale,
        diffusion_denoising_steps=diffusion_denoising_steps,
    )

    return AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=num_outputs,
        max_tokens=0,
        random=random,
        image_generation_parameters=image_generation_parameters,
    )


############################################################
# HEIM metrics


def get_core_heim_metric_specs() -> List[MetricSpec]:
    """Evaluate every image with these set of metrics."""
    return [
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.aesthetics_metrics.AestheticsMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.efficiency_metrics.EfficiencyMetric", args={}),
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.fractal_dimension_metric.FractalDimensionMetric",
            args={},
        ),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.nsfw_metrics.NSFWMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.nudity_metrics.NudityMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.watermark_metrics.WatermarkMetric", args={}),
    ] + get_basic_metric_specs(names=[])


def get_heim_bias_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.gender_metrics.GenderMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.skin_tone_metrics.SkinToneMetric", args={}),
    ]


def get_heim_detection_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.image_generation.detection_metrics.DetectionMetric", args={})]


def get_fid_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.fidelity_metrics.FidelityMetric", args={}),
    ]


def get_heim_reference_required_metric_specs(include_fidelity: bool = False) -> List[MetricSpec]:
    metrics: List[MetricSpec] = [
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.lpips_metrics."
            "LearnedPerceptualImagePatchSimilarityMetric",
            args={},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.multi_scale_ssim_metrics."
            "MultiScaleStructuralSimilarityIndexMeasureMetric",
            args={},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.psnr_metrics.PeakSignalToNoiseRatioMetric", args={}
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.uiqi_metrics.UniversalImageQualityIndexMetric", args={}
        ),
    ]
    if include_fidelity:
        metrics.extend(get_fid_metric_specs())
    return metrics


def get_heim_critique_metric_specs(
    include_aesthetics: bool = False,
    include_subject: bool = False,
    include_originality: bool = False,
    include_copyright: bool = False,
    num_examples: int = 10,
    num_respondents: int = 5,
    use_perturbed: bool = False,
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.image_critique_metrics.ImageCritiqueMetric",
            args={
                "include_alignment": True,  # Always ask about image-text alignment
                "include_aesthetics": include_aesthetics,
                "include_subject": include_subject,
                "include_originality": include_originality,
                "include_copyright": include_copyright,
                "num_examples": num_examples,
                "num_respondents": num_respondents,
                "use_perturbed": use_perturbed,
            },
        ),
    ]


def get_heim_photorealism_critique_metric_specs(
    num_examples: int = 100, num_respondents: int = 5, use_perturbed: bool = False
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.photorealism_critique_metrics."
            "PhotorealismCritiqueMetric",
            args={"num_examples": num_examples, "num_respondents": num_respondents, "use_perturbed": use_perturbed},
        ),
    ]


############################################################
# HEIM run specs


@run_spec_function("common_syntactic_processes")
def get_common_syntactic_processes_spec(phenomenon: str, run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation."
        "common_syntactic_processes_scenario.CommonSyntacticProcessesScenario",
        args={"phenomenon": phenomenon},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    run_spec_name: str = f"common_syntactic_processes:phenomenon={phenomenon}"
    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["common_syntactic_processes"],
    )


@run_spec_function("cub200")
def get_cub200_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.cub200_scenario.CUB200Scenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=1)

    metric_specs: List[MetricSpec] = get_heim_reference_required_metric_specs() + get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(include_aesthetics=True, include_subject=True, num_examples=10)

    return RunSpec(
        name="cub200",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["cub200"],
    )


@run_spec_function("daily_dalle")
def get_daily_dalle_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.daily_dalle_scenario.DailyDallEScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(
            include_aesthetics=True,
            include_subject=True,
            include_originality=True,
            num_examples=25,
        )

    return RunSpec(
        name="daily_dalle",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["daily_dalle"],
    )


@run_spec_function("demographic_stereotypes")
def get_demographic_stereotypes_spec(category: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation."
        "demographic_stereotypes_scenario.DemographicStereotypesScenario",
        args={"category": category},
    )

    # From https://arxiv.org/abs/2211.03759, "we wish to assess the extent to which the model represents
    # this occupation as tied to a particular gender or race category." So, for each occupation, they
    # compared to "reported percent of the occupation that self-identified as female and non-White."
    # Also, using "a photo of the face of X" generally generates clearer, more visible faces.
    adapter_spec = get_image_generation_adapter_spec(num_outputs=8)

    return RunSpec(
        name=f"demographic_stereotypes:category={category}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_heim_bias_metric_specs() + get_core_heim_metric_specs(),
        groups=["demographic_stereotypes"],
    )


@run_spec_function("detection")
def get_detection_spec(skill: str, run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.detection_scenario.DetectionScenario",
        args={"skill": skill},
    )

    adapter_spec: AdapterSpec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_heim_detection_metric_specs() + get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name=f"detection:skill={skill}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["detection"],
    )


@run_spec_function("draw_bench")
def get_draw_bench_spec(category: str, run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.draw_bench_scenario.DrawBenchScenario",
        args={"category": category},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    group: str
    if category in ["Colors", "Text", "Rare"]:
        group = "image_quality"
    elif category == "Reddit":
        group = "knowledge"
    elif category == "Misspellings":
        group = "robustness"
    else:
        group = "reasoning"

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    run_spec_name: str = f"draw_bench:category={category}"

    if run_human_eval:
        if category == "Reddit":
            metric_specs += get_heim_critique_metric_specs(num_examples=34)
        elif category in ["Colors", "Text", "Rare"]:
            metric_specs += get_heim_critique_metric_specs(
                include_aesthetics=True, include_subject=True, num_examples=10
            )
        else:
            metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[f"draw_bench_{group}"],
    )


@run_spec_function("i2p")
def get_i2p_spec(category: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.i2p_scenario.I2PScenario", args={"category": category}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=8)

    return RunSpec(
        name=f"i2p:category={category}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_core_heim_metric_specs(),
        groups=["i2p"],
    )


@run_spec_function("landing_page")
def get_landing_page_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.landing_page_scenario.LandingPageScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(
            include_aesthetics=True,
            include_subject=True,
            include_originality=True,
            num_examples=25,
        )

    return RunSpec(
        name="landing_page",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["landing_page"],
    )


@run_spec_function("logos")
def get_logos_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.logos_scenario.LogosScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(
            include_aesthetics=True,
            include_subject=True,
            include_originality=True,
            num_examples=25,
        )

    return RunSpec(
        name="logos",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["logos"],
    )


@run_spec_function("magazine_cover")
def get_magazine_cover_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.magazine_cover_scenario.MagazineCoverScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(
            include_aesthetics=True,
            include_subject=True,
            include_originality=True,
            num_examples=25,
        )

    return RunSpec(
        name="magazine_cover",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["magazine_cover"],
    )


@run_spec_function("mental_disorders")
def get_mental_disorders_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.mental_disorders_scenario.MentalDisordersScenario",
        args={},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=8)

    return RunSpec(
        name="mental_disorders",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_heim_bias_metric_specs() + get_core_heim_metric_specs(),
        groups=["mental_disorders"],
    )


@run_spec_function("mscoco")
def get_mscoco_spec(
    for_efficiency: bool = False,
    compute_fid: bool = False,
    run_human_eval: bool = False,
    num_human_examples: int = 100,
    use_perturbed: bool = False,
    skip_photorealism: bool = False,
    skip_subject: bool = False,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.mscoco_scenario.MSCOCOScenario", args={}
    )

    adapter_spec: AdapterSpec
    metric_specs: List[MetricSpec]
    run_spec_name: str

    if for_efficiency:
        adapter_spec = get_image_generation_adapter_spec(num_outputs=1)
        metric_specs = [
            MetricSpec(
                class_name="helm.benchmark.metrics.image_generation.denoised_runtime_metric.DenoisedRuntimeMetric",
                args={},
            ),
        ]
        run_spec_name = "mscoco_efficiency"
    elif compute_fid:
        adapter_spec = get_image_generation_adapter_spec(num_outputs=1)
        metric_specs = get_fid_metric_specs()
        run_spec_name = "mscoco_fid"
    else:
        adapter_spec = get_image_generation_adapter_spec(num_outputs=4)
        metric_specs = get_heim_reference_required_metric_specs() + get_core_heim_metric_specs()
        if run_human_eval:
            metric_specs += get_heim_critique_metric_specs(
                num_examples=num_human_examples,
                include_aesthetics=True,
                include_subject=not skip_subject,
                use_perturbed=use_perturbed,
            )
            if not skip_photorealism:
                metric_specs += get_heim_photorealism_critique_metric_specs(
                    num_examples=num_human_examples, use_perturbed=use_perturbed
                )
        run_spec_name = "mscoco"

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("paint_skills")
def get_paint_skills_spec(skill: str, run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.paint_skills_scenario.PaintSkillsScenario",
        args={"skill": skill},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    run_spec_name: str = f"paint_skills:skill={skill}"
    metric_specs: List[MetricSpec] = get_heim_reference_required_metric_specs() + get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["paint_skills"],
    )


@run_spec_function("parti_prompts")
def get_parti_prompts_spec(category: str, run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.parti_prompts_scenario.PartiPromptsScenario",
        args={"category": category},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    group: str
    if category == "Illustrations":
        group = "reasoning"
    elif category == "World":
        group = "knowledge"
    elif category == "Abstract":
        group = "extra"
    else:
        group = "image_quality"

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    run_spec_name: str = f"parti_prompts:category={category}"

    if run_human_eval:
        if category == "Illustrations":
            metric_specs += get_heim_critique_metric_specs(num_examples=10)
        elif category == "World":
            metric_specs += get_heim_critique_metric_specs(num_examples=34)
        else:
            metric_specs += get_heim_critique_metric_specs(
                include_aesthetics=True, include_subject=True, num_examples=10
            )

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[f"parti_prompts_{group}"],
    )


@run_spec_function("radiology")
def get_radiology_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.radiology_scenario.RadiologyScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    return RunSpec(
        name="radiology",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_core_heim_metric_specs(),
        groups=["radiology"],
    )


@run_spec_function("relational_understanding")
def get_relational_understanding_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation."
        "relational_understanding_scenario.RelationalUnderstandingScenario",
        args={},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name="relational_understanding",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["relational_understanding"],
    )


@run_spec_function("time_most_significant_historical_figures")
def get_time_most_significant_historical_figures_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.time_most_significant_historical_figures_scenario."
        "TIMEMostSignificantHistoricalFigures",
        args={},
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=34)

    return RunSpec(
        name="time_most_significant_historical_figures",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["time_most_significant_historical_figures"],
    )


@run_spec_function("winoground")
def get_winoground_spec(run_human_eval: bool = False) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.winoground_scenario.WinogroundScenario", args={}
    )

    adapter_spec = get_image_generation_adapter_spec(num_outputs=4)

    metric_specs: List[MetricSpec] = get_heim_reference_required_metric_specs() + get_core_heim_metric_specs()
    if run_human_eval:
        metric_specs += get_heim_critique_metric_specs(num_examples=10)

    return RunSpec(
        name="winoground",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["winoground"],
    )
