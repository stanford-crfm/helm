"""
Robustness Metrics for HELM

This module provides comprehensive robustness analysis metrics for evaluating
model stability, sensitivity, and adversarial robustness. These metrics help
understand how models perform under perturbations and variations.

Key Features:
- Sensitivity analysis to input perturbations
- Stability metrics across multiple trials
- Adversarial robustness indicators
- Consistency metrics
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import (
    Metric,
    MetricMetadata,
    MetricName,
    MetricResult,
    PerInstanceStats,
    add_context,
    get_unique_stat_by_name,
)
from helm.benchmark.metrics.metric_name import MetricContext
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.scenarios.scenario import Instance
from helm.common.hierarchical_logger import hlog
from helm.benchmark.augmentations.perturbation_description import PERTURBATION_ORIGINAL


class RobustnessMetric(Metric):
    """
    Comprehensive robustness metrics for language models.

    This metric computes:
    1. Sensitivity to perturbations (how much outputs change)
    2. Stability across trials (consistency of predictions)
    3. Adversarial robustness (performance under perturbations)
    4. Consistency metrics (agreement across variations)
    """

    def __init__(
        self,
        compute_sensitivity: bool = True,
        compute_stability: bool = True,
        compute_adversarial: bool = True,
        sensitivity_threshold: float = 0.1,
    ):
        """
        Initialize the robustness metric.

        Args:
            compute_sensitivity: Whether to compute sensitivity to perturbations
            compute_stability: Whether to compute stability across trials
            compute_adversarial: Whether to compute adversarial robustness
            sensitivity_threshold: Threshold for considering a change significant
        """
        self.compute_sensitivity = compute_sensitivity
        self.compute_stability = compute_stability
        self.compute_adversarial = compute_adversarial
        self.sensitivity_threshold = sensitivity_threshold

    def __repr__(self):
        return f"RobustnessMetric(compute_sensitivity={self.compute_sensitivity}, compute_stability={self.compute_stability})"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        """
        Compute robustness metrics across instances and perturbations.

        This requires comparing outputs across different perturbations and trials.
        """
        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}
        all_per_instance_stats: List[PerInstanceStats] = []

        # Group request states by instance
        instance_to_states: Dict[Instance, List] = defaultdict(list)
        for request_state in scenario_state.request_states:
            instance_to_states[request_state.instance].append(request_state)

        # Compute robustness metrics for each instance
        for instance, request_states in instance_to_states.items():
            instance_stats: List[Stat] = []

            # Compute stability across trials
            if self.compute_stability and adapter_spec.num_train_trials > 1:
                instance_stats.extend(self._compute_stability_metrics(instance, request_states))

            # Compute sensitivity to perturbations
            if self.compute_sensitivity:
                instance_stats.extend(self._compute_sensitivity_metrics(instance, request_states))

            # Compute adversarial robustness
            if self.compute_adversarial:
                instance_stats.extend(self._compute_adversarial_metrics(instance, request_states))

            if instance_stats:
                # Add context to stats
                context = MetricContext.from_instance(instance)
                for i, stat in enumerate(instance_stats):
                    instance_stats[i] = add_context(stat, context)

                # Aggregate instance stats
                for stat in instance_stats:
                    merge_stat(global_stats, stat)

                # Store per-instance stats
                if instance.id is not None:
                    all_per_instance_stats.append(
                        PerInstanceStats(
                            instance.id,
                            instance.perturbation,
                            0,  # train_trial_index
                            instance_stats,
                        )
                    )

        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def _compute_stability_metrics(
        self, instance: Instance, request_states: List
    ) -> List[Stat]:
        """Compute stability metrics across multiple trials."""
        stats: List[Stat] = []

        # Group by trial
        trial_outputs: Dict[int, List[str]] = defaultdict(list)
        trial_logprobs: Dict[int, List[float]] = defaultdict(list)

        for request_state in request_states:
            if request_state.result is None or len(request_state.result.completions) == 0:
                continue

            trial_idx = request_state.train_trial_index
            completion = request_state.result.completions[0]
            trial_outputs[trial_idx].append(completion.text)

            # Extract logprobs if available
            if completion.tokens:
                logprobs = [t.logprob for t in completion.tokens if t.logprob is not None]
                if logprobs:
                    trial_logprobs[trial_idx].append(np.mean(logprobs))

        if len(trial_outputs) < 2:
            return stats  # Need at least 2 trials for stability

        # Compute output consistency
        all_outputs = list(trial_outputs.values())
        unique_outputs = set()
        for outputs in all_outputs:
            unique_outputs.update(outputs)

        consistency_ratio = 1.0 - (len(unique_outputs) - 1) / (len(all_outputs) * len(all_outputs[0]) + 1e-10)
        stats.append(Stat(MetricName("output_consistency")).add(consistency_ratio))

        # Compute logprob stability
        if trial_logprobs:
            all_logprobs = [np.mean(logprobs) for logprobs in trial_logprobs.values() if logprobs]
            if len(all_logprobs) > 1:
                logprob_std = np.std(all_logprobs)
                logprob_mean = np.mean(all_logprobs)
                logprob_cv = logprob_std / (abs(logprob_mean) + 1e-10)
                stats.append(Stat(MetricName("logprob_stability")).add(1.0 / (1.0 + logprob_cv)))
                stats.append(Stat(MetricName("logprob_variance")).add(logprob_std**2))

        return stats

    def _compute_sensitivity_metrics(
        self, instance: Instance, request_states: List
    ) -> List[Stat]:
        """Compute sensitivity metrics to perturbations."""
        stats: List[Stat] = []

        # Find original and perturbed outputs
        original_states = [
            rs for rs in request_states
            if rs.instance.perturbation is None or rs.instance.perturbation.name == PERTURBATION_ORIGINAL
        ]
        perturbed_states = [
            rs for rs in request_states
            if rs.instance.perturbation is not None and rs.instance.perturbation.name != PERTURBATION_ORIGINAL
        ]

        if not original_states or not perturbed_states:
            return stats

        # Get original output
        if original_states[0].result is None or len(original_states[0].result.completions) == 0:
            return stats

        original_output = original_states[0].result.completions[0].text
        original_logprob = None
        if original_states[0].result.completions[0].tokens:
            logprobs = [t.logprob for t in original_states[0].result.completions[0].tokens if t.logprob is not None]
            if logprobs:
                original_logprob = np.mean(logprobs)

        # Compare with perturbed outputs
        output_changes = []
        logprob_changes = []

        for perturbed_state in perturbed_states:
            if perturbed_state.result is None or len(perturbed_state.result.completions) == 0:
                continue

            perturbed_output = perturbed_state.result.completions[0].text
            output_changed = 1.0 if perturbed_output != original_output else 0.0
            output_changes.append(output_changed)

            if original_logprob is not None:
                perturbed_logprobs = [
                    t.logprob for t in perturbed_state.result.completions[0].tokens
                    if t.logprob is not None
                ]
                if perturbed_logprobs:
                    perturbed_logprob = np.mean(perturbed_logprobs)
                    logprob_change = abs(perturbed_logprob - original_logprob)
                    logprob_changes.append(logprob_change)

        if output_changes:
            sensitivity_score = np.mean(output_changes)
            stats.append(Stat(MetricName("perturbation_sensitivity")).add(sensitivity_score))

        if logprob_changes:
            avg_logprob_change = np.mean(logprob_changes)
            stats.append(Stat(MetricName("logprob_sensitivity")).add(avg_logprob_change))

        return stats

    def _compute_adversarial_metrics(
        self, instance: Instance, request_states: List
    ) -> List[Stat]:
        """Compute adversarial robustness metrics."""
        stats: List[Stat] = []

        # Find original and worst-case perturbations
        original_states = [
            rs for rs in request_states
            if rs.instance.perturbation is None or rs.instance.perturbation.name == PERTURBATION_ORIGINAL
        ]
        worst_states = [
            rs for rs in request_states
            if rs.instance.perturbation is not None
            and "worst" in rs.instance.perturbation.name.lower()
        ]

        if not original_states:
            return stats

        # Get original performance
        if original_states[0].result is None or len(original_states[0].result.completions) == 0:
            return stats

        original_output = original_states[0].result.completions[0].text
        original_correct = 1.0 if any(ref.is_correct for ref in instance.references) else 0.0

        # Check if worst-case perturbations exist
        if worst_states:
            worst_correct = []
            for worst_state in worst_states:
                if worst_state.result is None or len(worst_state.result.completions) == 0:
                    continue
                # Check if output matches any correct reference
                worst_output = worst_state.result.completions[0].text
                worst_is_correct = 1.0 if any(
                    ref.output.text.strip() == worst_output.strip() for ref in instance.references if ref.is_correct
                ) else 0.0
                worst_correct.append(worst_is_correct)

            if worst_correct:
                adversarial_robustness = np.mean(worst_correct)
                robustness_drop = original_correct - adversarial_robustness
                stats.append(Stat(MetricName("adversarial_robustness")).add(adversarial_robustness))
                stats.append(Stat(MetricName("robustness_drop")).add(robustness_drop))

        return stats

    def get_metadata(self) -> List[MetricMetadata]:
        """Return metadata for all computed metrics."""
        return [
            MetricMetadata(
                name="output_consistency",
                display_name="Output Consistency",
                short_display_name="Consistency",
                description="Consistency of outputs across multiple trials. Higher values indicate more stable predictions.",
                lower_is_better=False,
                group="robustness",
            ),
            MetricMetadata(
                name="logprob_stability",
                display_name="Logprob Stability",
                short_display_name="Logprob Stability",
                description="Stability of log probabilities across trials. Higher values indicate more consistent confidence.",
                lower_is_better=False,
                group="robustness",
            ),
            MetricMetadata(
                name="logprob_variance",
                display_name="Logprob Variance",
                short_display_name="Logprob Var",
                description="Variance of log probabilities across trials. Lower values indicate more stability.",
                lower_is_better=True,
                group="robustness",
            ),
            MetricMetadata(
                name="perturbation_sensitivity",
                display_name="Perturbation Sensitivity",
                short_display_name="Sensitivity",
                description="Fraction of perturbations that change the output. Higher values indicate more sensitivity.",
                lower_is_better=True,
                group="robustness",
            ),
            MetricMetadata(
                name="logprob_sensitivity",
                display_name="Logprob Sensitivity",
                short_display_name="Logprob Sensitivity",
                description="Average change in log probability under perturbations. Lower values indicate more robustness.",
                lower_is_better=True,
                group="robustness",
            ),
            MetricMetadata(
                name="adversarial_robustness",
                display_name="Adversarial Robustness",
                short_display_name="Adv. Robustness",
                description="Accuracy under worst-case perturbations. Higher values indicate better adversarial robustness.",
                lower_is_better=False,
                group="robustness",
            ),
            MetricMetadata(
                name="robustness_drop",
                display_name="Robustness Drop",
                short_display_name="Robustness Drop",
                description="Drop in accuracy from original to worst-case perturbations. Lower values indicate better robustness.",
                lower_is_better=True,
                group="robustness",
            ),
        ]

