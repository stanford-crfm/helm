"""Run specs for Arabic Enterprise leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arabic_content_generation")
def get_arabic_content_generation_spec(category: str, annotator_type: str = "absolute") -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arabic_content_generation_scenario.ArabicContentGenerationScenario",
        args={"category": category},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Given the following lists of facts and style features, generate a professional business article in Modern Standard Arabic. Use all of the provided facts. Use only the provided facts, and do not introduce new facts. Respond only with the article.",
        max_tokens=1000,
        stop_sequences=[],
    )

    # annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.arabic_content_generation_annotator.ArabicContentGenerationAnnotator")]
    if annotator_type == "absolute":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_annotator.ArabicContentGenerationAnnotator"
            )
        ]
    elif annotator_type == "relative":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_relative_annotator.ArabicContentGenerationRelativeAnnotator"
            )
        ]
    elif annotator_type == "similarity":
        annotator_specs = [
            AnnotatorSpec(
                class_name="helm.benchmark.annotation.arabic_content_generation_similarity_annotator.ArabicContentGenerationSimilarityAnnotator"
            )
        ]
    else:
        raise ValueError(f"Unknown annotator_type {annotator_type}")
    metric_specs = get_basic_metric_specs([])

    return RunSpec(
        name=f"arabic_content_generation:category={category},annotator_type={annotator_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["arabic_content_generation"],
    )
