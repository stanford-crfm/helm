from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG
from sentence_transformers import SentenceTransformer


class SemanticSimilarityMetric(Metric):
    """Score metrics for LMKT semantic similarity measurement."""

    def __init__(self, similarity_fn_name: str = "cosine"):
        """
        Initialize the SemanticSimilarityMetric with a SentenceTransformer model.
        :param similarity_fn_name: The name of the similarity function to use.
        Available options are "dot", "cosine", "manhattan" and "euclidean".
        """
        super().__init__()

        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", similarity_fn_name=similarity_fn_name)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:

        assert request_state.result

        completions = [c.text for c in request_state.result.completions]
        completion_embeddings = self.model.encode(completions)

        references = [r.output.text for r in request_state.instance.references if CORRECT_TAG in r.tags]
        reference_embeddings = self.model.encode(references)

        similarities = self.model.similarity(completion_embeddings, reference_embeddings)
        avg_similarity = similarities.mean().item()

        return [
            Stat(MetricName("semantic_similarity")).add(avg_similarity),
        ]
