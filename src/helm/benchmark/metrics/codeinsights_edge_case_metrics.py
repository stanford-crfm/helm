from typing import List
import re

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.codeinsights_code_evaluation_metrics import CodeInsightsCodeEvaluationMetric


class UnittestAlignmentMetric(Metric):
    """
    Compare LLM unit-test results with the student’s correctness pattern.

    Adds:
        • functional_correctness (pass-rate)
        • edge_case_slip_match   (binary 0/1)
    """

    # ------------------------------------------------------------------ I#
    #   HELM entry-point                                                 #
    # ------------------------------------------------------------------ #
    def evaluate_generation(  # HELM entry-point
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # ------------------------------------------------------------------
        # 1. Parse the model’s answer --------------------------------------
        # ------------------------------------------------------------------
        default_stat = Stat(MetricName("unittest_alignment")).add(0.0)

        if not request_state.result or not request_state.result.completions:
            # No output → automatic miss
            return [default_stat]

        raw_output: str = request_state.result.completions[0].text.strip()

        # Extract the *first* integer we see (robust to whitespace / newlines)
        match = re.search(r"-?\d+", raw_output)
        if match is None:
            # Model didn’t emit an integer → miss
            return [default_stat]

        try:
            predicted_index: int = int(match.group())
        except ValueError:
            # Shouldn’t happen, but be safe
            return [default_stat]

        # ------------------------------------------------------------------
        # 2. Retrieve ground-truth failure index ---------------------------
        # ------------------------------------------------------------------
        extra = getattr(request_state.instance, "extra_data", {}) or {}
        correctness_pattern: List[int] = extra.get("student_correctness_pattern", [])

        # Indices where the student failed (value == 0)
        failed_indices: List[int] = [i for i, v in enumerate(correctness_pattern) if v == 0]

        # If we don’t have exactly one failing test, treat as miss
        if len(failed_indices) != 1:
            return [default_stat]

        actual_index: int = failed_indices[0]

        # ------------------------------------------------------------------
        # 3. Compare & return ---------------------------------------------
        # ------------------------------------------------------------------
        alignment_score = 1.0 if predicted_index == actual_index else 0.0
        return [Stat(MetricName("unittest_alignment")).add(alignment_score)]


class CodeInsightsUnittestAlignmentMetric(CodeInsightsCodeEvaluationMetric):
    """unit-test alignment (with new metrics)."""

    def __init__(self, use_codebert: bool = True):
        super().__init__(use_codebert=use_codebert)
        self.alignment_metric = UnittestAlignmentMetric()

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ):
        stats = self.alignment_metric.evaluate_generation(adapter_spec, request_state, metric_service, eval_cache_path)
        return stats
