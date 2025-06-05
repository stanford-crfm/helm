import itertools
from typing import List

from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs


def get_vietnamese_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.melt_toxicity_metric.VietnameseToxicityMetric", args={}),
    ]


def get_vietnamese_bias_metric_specs() -> List[MetricSpec]:
    demographic_categories = ["race", "gender"]
    target_categories = ["adjective", "profession"]
    cross_dem_target = itertools.product(demographic_categories, target_categories)

    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.melt_bias_metric.VietnameseBiasMetric",
            args={"mode": "associations", "demographic_category": dem, "target_category": tgt},
        )
        for dem, tgt in cross_dem_target
    ] + [
        MetricSpec(
            class_name="helm.benchmark.metrics.melt_bias_metric.VietnameseBiasMetric",
            args={"mode": "representation", "demographic_category": dem},
        )
        for dem in demographic_categories
    ]


def get_vietnamese_generative_harms_metric_specs(
    include_basic_metrics: bool = False, include_generative_harms_metrics: bool = False
) -> List[MetricSpec]:
    metric_specs: List[MetricSpec] = []
    if include_basic_metrics:
        metric_specs.extend(get_basic_metric_specs([]))
    if include_generative_harms_metrics:
        metric_specs.extend(get_vietnamese_bias_metric_specs())
        metric_specs.extend(get_vietnamese_toxicity_metric_specs())
    return metric_specs
