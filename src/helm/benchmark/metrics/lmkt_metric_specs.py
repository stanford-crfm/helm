from typing import List

from helm.benchmark.metrics.metric import MetricSpec


def get_semantic_similarity_metric_specs(similarity_fn_name: str = None) -> List[MetricSpec]:
    if similarity_fn_name is None:
        similarity_fn_name = "cosine"  # Default similarity function
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.lmkt_metrics.SemanticSimilarityMetric",
            args={"similarity_fn_name": similarity_fn_name},
        ),
    ]
