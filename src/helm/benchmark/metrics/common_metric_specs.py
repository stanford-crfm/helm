import itertools
from typing import Any, Dict, List, Optional

from helm.benchmark.metrics.metric import MetricSpec


def get_basic_generation_metric_specs(names: List[str]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": names}),
    ]


def get_basic_reference_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicReferenceMetric", args={}),
    ]


def get_generic_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric", args={}),
    ]


def get_basic_metric_specs(names: List[str]) -> List[MetricSpec]:
    return get_basic_generation_metric_specs(names) + get_basic_reference_metric_specs() + get_generic_metric_specs()


def get_exact_match_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "prefix_exact_match", "quasi_prefix_exact_match"]
    )


def get_f1_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match", "f1_score"])


def get_language_modeling_metric_specs(names: List[str]) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.language_modeling_metrics.LanguageModelingMetric", args={"names": names}
        )
    ]


def get_classification_metric_specs(
    labels: Optional[List[str]] = None, delimiter: Optional[str] = None
) -> List[MetricSpec]:
    extra_args: Dict[str, Any] = {}
    if labels:
        extra_args["labels"] = labels
    if delimiter:
        extra_args["delimiter"] = delimiter
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.classification_metrics.ClassificationMetric",
            args=extra_args,
        )
    ]


def get_multiple_choice_classification_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={}
        )
    ]


def get_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.toxicity_metrics.ToxicityMetric", args={}),
    ]


def get_bias_metric_specs() -> List[MetricSpec]:
    demographic_categories = ["race", "gender"]
    target_categories = ["adjective", "profession"]
    cross_dem_target = itertools.product(demographic_categories, target_categories)

    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.bias_metrics.BiasMetric",
            args={"mode": "associations", "demographic_category": dem, "target_category": tgt},
        )
        for dem, tgt in cross_dem_target
    ] + [
        MetricSpec(
            class_name="helm.benchmark.metrics.bias_metrics.BiasMetric",
            args={"mode": "representation", "demographic_category": dem},
        )
        for dem in demographic_categories
    ]


def get_generative_harms_metric_specs(
    include_basic_metrics: bool = False, include_generative_harms_metrics: bool = False
) -> List[MetricSpec]:
    metric_specs: List[MetricSpec] = []
    if include_basic_metrics:
        metric_specs.extend(get_basic_metric_specs([]))
    if include_generative_harms_metrics:
        metric_specs.extend(get_bias_metric_specs())
        metric_specs.extend(get_toxicity_metric_specs())
    return metric_specs


def get_summarization_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args=args)
    ] + get_basic_metric_specs([])


def get_summarization_critique_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.summarization_critique_metrics.SummarizationCritiqueMetric",
            args={"num_respondents": num_respondents},
        )
    ]


def get_numeracy_metric_specs(run_solver: bool = False) -> List[MetricSpec]:
    metric_specs: List[MetricSpec] = get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "absolute_value_difference"]
    )

    # The solvers are slow to run so make them skippable
    if run_solver:
        metric_specs += [
            MetricSpec(class_name="helm.benchmark.metrics.numeracy_metrics.DistanceMetric", args={}),
        ]
    return metric_specs


def get_copyright_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "longest_common_prefix_length"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "edit_distance"},
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.copyright_metrics.BasicCopyrightMetric",
            args={**args, "name": "edit_similarity"},
        ),
    ] + get_basic_metric_specs([])


def get_disinformation_metric_specs(args: Optional[Dict] = None) -> List[MetricSpec]:
    if args is None:
        args = {}
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationHumanEvalMetrics", args={**args}
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric", args={"name": "self_bleu"}
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric",
            args={"name": "monte_carlo_entropy"},
        ),
    ] + get_basic_metric_specs([])


def get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"])


def get_uncertainty_quantification_metric_specs(
    num_bins: int = 10,
    confidence_levels: Optional[List[float]] = None,
    compute_uncertainty_decomposition: bool = True,
) -> List[MetricSpec]:
    """Get metric specs for uncertainty quantification metrics."""
    args: Dict[str, Any] = {
        "num_bins": num_bins,
        "compute_uncertainty_decomposition": compute_uncertainty_decomposition,
    }
    if confidence_levels is not None:
        args["confidence_levels"] = confidence_levels
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.uncertainty_quantification_metrics.UncertaintyQuantificationMetric",
            args=args,
        )
    ]


def get_robustness_metric_specs(
    compute_sensitivity: bool = True,
    compute_stability: bool = True,
    compute_adversarial: bool = True,
) -> List[MetricSpec]:
    """Get metric specs for robustness metrics."""
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.robustness_metrics.RobustnessMetric",
            args={
                "compute_sensitivity": compute_sensitivity,
                "compute_stability": compute_stability,
                "compute_adversarial": compute_adversarial,
            },
        )
    ]


def get_cross_model_consistency_metric_specs(
    compute_agreement: bool = True,
    compute_consensus: bool = True,
    compute_disagreement: bool = True,
) -> List[MetricSpec]:
    """Get metric specs for cross-model consistency metrics."""
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cross_model_consistency_metrics.CrossModelConsistencyMetric",
            args={
                "compute_agreement": compute_agreement,
                "compute_consensus": compute_consensus,
                "compute_disagreement": compute_disagreement,
            },
        )
    ]


def get_research_metrics_specs() -> List[MetricSpec]:
    """
    Get all research-level metrics (uncertainty, robustness, cross-model consistency).
    
    This is a convenience function that returns all research metrics together.
    """
    return (
        get_uncertainty_quantification_metric_specs()
        + get_robustness_metric_specs()
        + get_cross_model_consistency_metric_specs()
    )