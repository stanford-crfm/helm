"""
Uncertainty Quantification Metrics for HELM

This module provides comprehensive uncertainty quantification metrics for evaluating
model confidence, prediction reliability, and uncertainty separation. These metrics
are particularly valuable for research on model calibration, robustness, and
trustworthy AI systems.

Key Features:
- Entropy-based uncertainty measures
- Prediction interval coverage
- Aleatoric vs Epistemic uncertainty separation
- Confidence calibration metrics
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.special  # type: ignore
import scipy.stats  # type: ignore

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import (
    Metric,
    MetricMetadata,
    MetricName,
    MetricResult,
    PerInstanceStats,
    add_context,
    get_unique_stat_by_name,
)
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.scenarios.scenario import Instance
from helm.common.hierarchical_logger import hlog
from helm.common.request import Token


class UncertaintyQuantificationMetric(Metric):
    """
    Comprehensive uncertainty quantification metrics for language models.

    This metric computes:
    1. Prediction entropy (Shannon entropy of predicted probabilities)
    2. Confidence intervals and coverage
    3. Uncertainty decomposition (aleatoric vs epistemic)
    4. Reliability metrics

    These metrics are computed from token-level log probabilities and are
    particularly useful for understanding model confidence and reliability.
    """

    def __init__(
        self,
        num_bins: int = 10,
        confidence_levels: Optional[List[float]] = None,
        compute_uncertainty_decomposition: bool = True,
    ):
        """
        Initialize the uncertainty quantification metric.

        Args:
            num_bins: Number of bins for entropy-based metrics
            confidence_levels: List of confidence levels (e.g., [0.8, 0.9, 0.95]) for coverage metrics
            compute_uncertainty_decomposition: Whether to compute aleatoric/epistemic uncertainty
        """
        self.num_bins = num_bins
        self.confidence_levels = confidence_levels or [0.8, 0.9, 0.95]
        self.compute_uncertainty_decomposition = compute_uncertainty_decomposition

    def __repr__(self):
        return f"UncertaintyQuantificationMetric(num_bins={self.num_bins}, confidence_levels={self.confidence_levels})"

    def evaluate_generation(
        self,
        adapter_spec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Compute uncertainty metrics for a single generation."""
        stats: List[Stat] = []

        if request_state.result is None or len(request_state.result.completions) == 0:
            return stats

        completion = request_state.result.completions[0]
        tokens: List[Token] = completion.tokens

        if not tokens or not any(token.logprob is not None for token in tokens):
            hlog("Skipping uncertainty metrics: no logprobs available")
            return stats

        # Extract log probabilities
        logprobs = [token.logprob for token in tokens if token.logprob is not None]
        if not logprobs:
            return stats

        # Compute basic uncertainty metrics
        stats.extend(self._compute_entropy_metrics(logprobs))
        stats.extend(self._compute_confidence_metrics(logprobs))
        stats.extend(self._compute_variance_metrics(logprobs))

        # Compute uncertainty decomposition if enabled
        if self.compute_uncertainty_decomposition and len(logprobs) > 1:
            stats.extend(self._compute_uncertainty_decomposition(logprobs))

        return stats

    def _compute_entropy_metrics(self, logprobs: List[float]) -> List[Stat]:
        """Compute entropy-based uncertainty metrics."""
        stats: List[Stat] = []

        # Convert logprobs to probabilities (normalized)
        probs = np.array([math.exp(lp) for lp in logprobs])
        probs = probs / (probs.sum() + 1e-10)  # Normalize

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        stats.append(Stat(MetricName("prediction_entropy")).add(entropy))

        # Max probability (confidence)
        max_prob = np.max(probs)
        stats.append(Stat(MetricName("max_probability")).add(max_prob))

        # Entropy ratio (normalized entropy)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        entropy_ratio = entropy / (max_entropy + 1e-10)
        stats.append(Stat(MetricName("entropy_ratio")).add(entropy_ratio))

        # Effective number of outcomes (exponential of entropy)
        effective_outcomes = math.exp(entropy) if entropy > 0 else 1.0
        stats.append(Stat(MetricName("effective_outcomes")).add(effective_outcomes))

        return stats

    def _compute_confidence_metrics(self, logprobs: List[float]) -> List[Stat]:
        """Compute confidence interval and coverage metrics."""
        stats: List[Stat] = []

        # Convert to probabilities
        probs = np.array([math.exp(lp) for lp in logprobs])
        probs = probs / (probs.sum() + 1e-10)

        # Sort probabilities in descending order
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)

        # Compute coverage for different confidence levels
        for conf_level in self.confidence_levels:
            # Find how many top outcomes are needed to reach confidence level
            coverage_count = np.searchsorted(cumulative_probs, conf_level, side="right") + 1
            coverage_count = min(coverage_count, len(sorted_probs))
            stats.append(
                Stat(MetricName(f"coverage_at_{int(conf_level * 100)}")).add(coverage_count / len(sorted_probs))
            )

        # Top-k confidence (k=1, 2, 3)
        for k in [1, 2, 3]:
            if len(sorted_probs) >= k:
                top_k_conf = np.sum(sorted_probs[:k])
                stats.append(Stat(MetricName(f"top_{k}_confidence")).add(top_k_conf))

        return stats

    def _compute_variance_metrics(self, logprobs: List[float]) -> List[Stat]:
        """Compute variance-based uncertainty metrics."""
        stats: List[Stat] = []

        # Convert to probabilities
        probs = np.array([math.exp(lp) for lp in logprobs])
        probs = probs / (probs.sum() + 1e-10)

        # Variance of probabilities
        prob_variance = np.var(probs)
        stats.append(Stat(MetricName("probability_variance")).add(prob_variance))

        # Coefficient of variation
        prob_mean = np.mean(probs)
        coeff_variation = np.sqrt(prob_variance) / (prob_mean + 1e-10)
        stats.append(Stat(MetricName("coefficient_variation")).add(coeff_variation))

        # Gini coefficient (measure of inequality)
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_probs)) / (n * np.sum(sorted_probs) + 1e-10) - (
            n + 1
        ) / n
        stats.append(Stat(MetricName("gini_coefficient")).add(gini))

        return stats

    def _compute_uncertainty_decomposition(self, logprobs: List[float]) -> List[Stat]:
        """
        Decompose uncertainty into aleatoric (data) and epistemic (model) components.

        This is a simplified decomposition based on the variance of predictions.
        """
        stats: List[Stat] = []

        # Convert to probabilities
        probs = np.array([math.exp(lp) for lp in logprobs])
        probs = probs / (probs.sum() + 1e-10)

        # Aleatoric uncertainty: entropy of the mean prediction
        mean_prob = np.mean(probs)
        aleatoric_entropy = -mean_prob * math.log(mean_prob + 1e-10) - (1 - mean_prob) * math.log(
            1 - mean_prob + 1e-10
        )
        stats.append(Stat(MetricName("aleatoric_entropy")).add(aleatoric_entropy))

        # Epistemic uncertainty: variance across predictions
        epistemic_uncertainty = np.var(probs)
        stats.append(Stat(MetricName("epistemic_uncertainty")).add(epistemic_uncertainty))

        # Total uncertainty (entropy)
        total_entropy = -np.sum(probs * np.log(probs + 1e-10))
        stats.append(Stat(MetricName("total_entropy")).add(total_entropy))

        # Uncertainty ratio
        if total_entropy > 0:
            aleatoric_ratio = aleatoric_entropy / (total_entropy + 1e-10)
            stats.append(Stat(MetricName("aleatoric_ratio")).add(aleatoric_ratio))

        return stats

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive aggregate uncertainty metrics."""
        derived_stats: List[Stat] = []

        # Compute average entropy across instances
        entropy_stat = get_unique_stat_by_name(stats_dict.values(), "prediction_entropy")
        if entropy_stat is not None and entropy_stat.count > 0:
            avg_entropy = entropy_stat.mean
            if avg_entropy is not None:
                derived_stats.append(Stat(MetricName("avg_prediction_entropy")).add(avg_entropy))

        return derived_stats

    def get_metadata(self) -> List[MetricMetadata]:
        """Return metadata for all computed metrics."""
        metadata = [
            MetricMetadata(
                name="prediction_entropy",
                display_name="Prediction Entropy",
                short_display_name="Entropy",
                description="Shannon entropy of the predicted probability distribution. Higher values indicate more uncertainty.",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="max_probability",
                display_name="Max Probability",
                short_display_name="Max Prob",
                description="Maximum probability assigned to any outcome. Higher values indicate higher confidence.",
                lower_is_better=False,
                group="uncertainty",
            ),
            MetricMetadata(
                name="entropy_ratio",
                display_name="Entropy Ratio",
                short_display_name="Entropy Ratio",
                description="Normalized entropy (entropy / max_entropy). Ranges from 0 (certain) to 1 (uniform).",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="effective_outcomes",
                display_name="Effective Outcomes",
                short_display_name="Eff. Outcomes",
                description="Exponential of entropy, representing the effective number of distinct outcomes.",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="probability_variance",
                display_name="Probability Variance",
                short_display_name="Prob Var",
                description="Variance of the probability distribution. Higher values indicate more spread.",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="coefficient_variation",
                display_name="Coefficient of Variation",
                short_display_name="CV",
                description="Standard deviation divided by mean probability. Measures relative variability.",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="gini_coefficient",
                display_name="Gini Coefficient",
                short_display_name="Gini",
                description="Measure of inequality in probability distribution. 0 = uniform, 1 = concentrated.",
                lower_is_better=None,
                group="uncertainty",
            ),
            MetricMetadata(
                name="aleatoric_entropy",
                display_name="Aleatoric Entropy",
                short_display_name="Aleatoric",
                description="Uncertainty due to inherent randomness in the data (data uncertainty).",
                lower_is_better=None,
                group="uncertainty_decomposition",
            ),
            MetricMetadata(
                name="epistemic_uncertainty",
                display_name="Epistemic Uncertainty",
                short_display_name="Epistemic",
                description="Uncertainty due to model limitations (model uncertainty).",
                lower_is_better=None,
                group="uncertainty_decomposition",
            ),
            MetricMetadata(
                name="total_entropy",
                display_name="Total Entropy",
                short_display_name="Total",
                description="Total uncertainty (aleatoric + epistemic).",
                lower_is_better=None,
                group="uncertainty_decomposition",
            ),
            MetricMetadata(
                name="aleatoric_ratio",
                display_name="Aleatoric Ratio",
                short_display_name="Aleatoric Ratio",
                description="Proportion of total uncertainty that is aleatoric (data-driven).",
                lower_is_better=None,
                group="uncertainty_decomposition",
            ),
        ]

        # Add coverage metrics
        for conf_level in self.confidence_levels:
            metadata.append(
                MetricMetadata(
                    name=f"coverage_at_{int(conf_level * 100)}",
                    display_name=f"Coverage at {int(conf_level * 100)}%",
                    short_display_name=f"Cov@{int(conf_level * 100)}%",
                    description=f"Fraction of outcomes needed to achieve {int(conf_level * 100)}% confidence.",
                    lower_is_better=None,
                    group="uncertainty",
                )
            )

        # Add top-k confidence metrics
        for k in [1, 2, 3]:
            metadata.append(
                MetricMetadata(
                    name=f"top_{k}_confidence",
                    display_name=f"Top-{k} Confidence",
                    short_display_name=f"Top-{k}",
                    description=f"Sum of probabilities for top {k} outcomes.",
                    lower_is_better=False,
                    group="uncertainty",
                )
            )

        return metadata

