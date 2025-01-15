from typing import List, Dict, Any
from helm.benchmark.metrics.metric import Metric, Stat, MetricName
from helm.benchmark.metrics.basic_metrics import compute_reference_metrics
from helm.benchmark.adapter import AdapterSpec, RequestState
from helm.common.general import asdict
from helm.common.object_spec import MetricService
from bert_score import BERTScorer

class ACIBenchMetric(Metric):
    """
    Metric for evaluating the ACI Bench dataset, assessing summarization quality of generated clinical notes.

    This implementation calculates:
    - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    - BLEURT (if available in the MetricService)
    - BERTScore
    - Other basic metrics provided by the HELM framework.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the metric with optional device specification for BERTScore.

        Args:
            device: Device to run BERTScore (e.g., "cpu" or "cuda").
        """
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, device=device
        )

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate a single generation against the reference labels.

        Args:
            adapter_spec: Adapter specification for the task.
            request_state: The request state containing the generated output.
            metric_service: Service to compute metrics like ROUGE and BLEURT.
            eval_cache_path: Path to cache evaluation results (not used here).

        Returns:
            List of Stat objects representing the computed metrics.
        """
        # Ensure there are references for the instance
        if not request_state.instance.references:
            raise ValueError("No references provided for the instance.")

        # Extract the generated output
        if not request_state.result.completions:
            raise ValueError("No completions provided for the request state.")

        generated_text = request_state.result.completions[0].text
        reference_texts = [ref.output.text for ref in request_state.instance.references]

        # Compute reference-based metrics (e.g., ROUGE, BLEURT)
        stats: List[Stat] = compute_reference_metrics(
            ["rouge1", "rouge2", "rougeL", "bleurt"],
            adapter_spec,
            request_state,
            metric_service,
        )

        # Compute BERTScore
        bert_scores = self._compute_bert_score(reference_texts, generated_text)
        stats.extend([Stat(MetricName(name)).add(value) for name, value in bert_scores.items()])

        return stats

    def _compute_bert_score(self, references: List[str], prediction: str) -> Dict[str, float]:
        """
        Compute BERTScore for the given prediction and references.

        Args:
            references: List of reference texts.
            prediction: Generated text to evaluate.

        Returns:
            A dictionary containing BERTScore precision, recall, and F1.
        """
        p, r, f = self.bert_scorer.score([prediction], references)
        return {
            "bertscore-precision": p.mean().item(),
            "bertscore-recall": r.mean().item(),
            "bertscore-f1": f.mean().item(),
        }

    def compute(self, stats: List[Stat], **kwargs: Any) -> Dict[str, float]:
        """
        Aggregate statistics to compute final metrics.

        Args:
            stats: List of Stat objects containing raw scores for each instance.

        Returns:
            A dictionary of aggregated metrics.
        """
        aggregated_metrics: Dict[str, float] = {}
        metric_groups: Dict[str, List[float]] = {}

        # Group metrics by name
        for stat in stats:
            metric_name = stat.name
            if metric_name not in metric_groups:
                metric_groups[metric_name] = []
            metric_groups[metric_name].append(stat.value)

        # Aggregate metrics (e.g., compute mean)
        for metric_name, values in metric_groups.items():
            aggregated_metrics[metric_name] = sum(values) / len(values)

        return aggregated_metrics
