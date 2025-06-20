from typing import List

from helm.benchmark.metrics.metric import MetricSpec


def get_semantic_similarity_metric_specs(similarity_fn_name: str = "cosine") -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.lmkt_metrics.SemanticSimilarityMetric",
            args={"similarity_fn_name": similarity_fn_name},
        ),
    ]
