"""
Cross-Model Consistency Metrics for HELM

This module provides metrics for analyzing consistency and agreement between
different models on the same evaluation instances. This is valuable for
understanding model consensus, disagreement patterns, and ensemble potential.

Key Features:
- Inter-model agreement metrics
- Consensus-based confidence scoring
- Model disagreement analysis
- Ensemble potential indicators
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import (
    MetricInterface,
    MetricMetadata,
    MetricName,
    MetricResult,
    PerInstanceStats,
    add_context,
)
from helm.benchmark.metrics.metric_name import MetricContext
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.scenarios.scenario import Instance
from helm.common.hierarchical_logger import hlog


class CrossModelConsistencyMetric(MetricInterface):
    """
    Metrics for analyzing consistency and agreement across multiple models.

    This metric computes:
    1. Inter-model agreement (how often models agree)
    2. Consensus confidence (confidence when models agree)
    3. Disagreement patterns (when and why models disagree)
    4. Ensemble potential (benefit of combining models)
    """

    def __init__(
        self,
        compute_agreement: bool = True,
        compute_consensus: bool = True,
        compute_disagreement: bool = True,
        agreement_threshold: float = 0.5,
    ):
        """
        Initialize the cross-model consistency metric.

        Args:
            compute_agreement: Whether to compute inter-model agreement
            compute_consensus: Whether to compute consensus-based metrics
            compute_disagreement: Whether to analyze disagreement patterns
            agreement_threshold: Threshold for considering models in agreement
        """
        self.compute_agreement = compute_agreement
        self.compute_consensus = compute_consensus
        self.compute_disagreement = compute_disagreement
        self.agreement_threshold = agreement_threshold

    def __repr__(self):
        return f"CrossModelConsistencyMetric(compute_agreement={self.compute_agreement})"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        """
        Compute cross-model consistency metrics.

        Note: This metric requires results from multiple models on the same instances.
        It should be run as a post-processing step after evaluating multiple models.
        """
        global_stats: Dict[MetricName, Stat] = {}
        all_per_instance_stats: List[PerInstanceStats] = []

        # Group request states by instance
        instance_to_states: Dict[Instance, List] = defaultdict(list)
        for request_state in scenario_state.request_states:
            instance_to_states[request_state.instance].append(request_state)

        # Group by model (extract from adapter_spec or request_state)
        # For now, we'll assume we can identify models from the scenario_state
        model_to_states: Dict[str, List] = defaultdict(list)
        for request_state in scenario_state.request_states:
            model_name = scenario_state.adapter_spec.model_deployment
            model_to_states[model_name].append(request_state)

        if len(model_to_states) < 2:
            hlog("CrossModelConsistencyMetric requires at least 2 models. Skipping.")
            return MetricResult([], [])

        # Compute metrics for each instance
        for instance, instance_states in instance_to_states.items():
            instance_stats: List[Stat] = []

            # Group instance states by model
            model_outputs: Dict[str, List[str]] = defaultdict(list)
            model_logprobs: Dict[str, List[float]] = defaultdict(list)
            model_correct: Dict[str, List[bool]] = defaultdict(list)

            for state in instance_states:
                if state.result is None or len(state.result.completions) == 0:
                    continue

                model_name = scenario_state.adapter_spec.model_deployment
                completion = state.result.completions[0]
                model_outputs[model_name].append(completion.text)

                # Extract logprobs
                if completion.tokens:
                    logprobs = [t.logprob for t in completion.tokens if t.logprob is not None]
                    if logprobs:
                        model_logprobs[model_name].append(np.mean(logprobs))

                # Check correctness
                is_correct = any(
                    ref.output.text.strip() == completion.text.strip()
                    for ref in instance.references
                    if ref.is_correct
                )
                model_correct[model_name].append(is_correct)

            if len(model_outputs) < 2:
                continue

            # Compute agreement metrics
            if self.compute_agreement:
                instance_stats.extend(self._compute_agreement_metrics(model_outputs, model_correct))

            # Compute consensus metrics
            if self.compute_consensus:
                instance_stats.extend(self._compute_consensus_metrics(model_outputs, model_logprobs, model_correct))

            # Compute disagreement metrics
            if self.compute_disagreement:
                instance_stats.extend(self._compute_disagreement_metrics(model_outputs, model_correct))

            if instance_stats:
                # Add context
                context = MetricContext.from_instance(instance)
                for i, stat in enumerate(instance_stats):
                    instance_stats[i] = add_context(stat, context)

                # Aggregate
                for stat in instance_stats:
                    merge_stat(global_stats, stat)

                # Store per-instance
                if instance.id is not None:
                    all_per_instance_stats.append(
                        PerInstanceStats(instance.id, instance.perturbation, 0, instance_stats)
                    )

        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def _compute_agreement_metrics(
        self, model_outputs: Dict[str, List[str]], model_correct: Dict[str, List[bool]]
    ) -> List[Stat]:
        """Compute inter-model agreement metrics."""
        stats: List[Stat] = []

        # Get unique outputs per model
        model_unique_outputs: Dict[str, Set[str]] = {
            model: set(outputs) for model, outputs in model_outputs.items()
        }

        # Compute pairwise agreement
        models = list(model_unique_outputs.keys())
        agreements = []
        for i, model1 in enumerate(models):
            for model2 in models[i + 1 :]:
                outputs1 = model_unique_outputs[model1]
                outputs2 = model_unique_outputs[model2]
                # Agreement: fraction of outputs that appear in both
                intersection = len(outputs1.intersection(outputs2))
                union = len(outputs1.union(outputs2))
                agreement = intersection / (union + 1e-10)
                agreements.append(agreement)

        if agreements:
            avg_agreement = np.mean(agreements)
            stats.append(Stat(MetricName("inter_model_agreement")).add(avg_agreement))

        # Compute correctness agreement
        if model_correct:
            model_accuracy: Dict[str, float] = {
                model: np.mean(corrects) for model, corrects in model_correct.items()
            }
            if len(model_accuracy) > 1:
                accuracies = list(model_accuracy.values())
                accuracy_std = np.std(accuracies)
                stats.append(Stat(MetricName("accuracy_consistency")).add(1.0 / (1.0 + accuracy_std)))

        return stats

    def _compute_consensus_metrics(
        self,
        model_outputs: Dict[str, List[str]],
        model_logprobs: Dict[str, List[float]],
        model_correct: Dict[str, List[bool]],
    ) -> List[Stat]:
        """Compute consensus-based metrics."""
        stats: List[Stat] = []

        # Find consensus outputs (outputs that appear in multiple models)
        all_outputs = []
        for outputs in model_outputs.values():
            all_outputs.extend(outputs)

        from collections import Counter

        output_counts = Counter(all_outputs)
        consensus_outputs = {output: count for output, count in output_counts.items() if count > 1}

        if consensus_outputs:
            consensus_ratio = len(consensus_outputs) / (len(set(all_outputs)) + 1e-10)
            stats.append(Stat(MetricName("consensus_ratio")).add(consensus_ratio))

            # Consensus confidence (average logprob for consensus outputs)
            if model_logprobs:
                consensus_logprobs = []
                for model, logprobs in model_logprobs.items():
                    outputs = model_outputs[model]
                    for i, output in enumerate(outputs):
                        if output in consensus_outputs and i < len(logprobs):
                            consensus_logprobs.append(logprobs[i])

                if consensus_logprobs:
                    avg_consensus_logprob = np.mean(consensus_logprobs)
                    stats.append(Stat(MetricName("consensus_confidence")).add(avg_consensus_logprob))

        return stats

    def _compute_disagreement_metrics(
        self, model_outputs: Dict[str, List[str]], model_correct: Dict[str, List[bool]]
    ) -> List[Stat]:
        """Compute disagreement analysis metrics."""
        stats: List[Stat] = []

        # Compute disagreement rate
        all_unique_outputs = set()
        for outputs in model_outputs.values():
            all_unique_outputs.update(outputs)

        num_models = len(model_outputs)
        if num_models > 1:
            # Disagreement: when models produce different outputs
            disagreement_rate = 1.0 - (len(all_unique_outputs) - num_models + 1) / (
                len(all_unique_outputs) + 1e-10
            )
            stats.append(Stat(MetricName("disagreement_rate")).add(disagreement_rate))

        # Compute correctness disagreement
        if model_correct:
            # Check if all models agree on correctness
            all_correct = []
            for corrects in model_correct.values():
                all_correct.extend(corrects)

            if len(set(all_correct)) > 1:
                correctness_disagreement = 1.0
            else:
                correctness_disagreement = 0.0
            stats.append(Stat(MetricName("correctness_disagreement")).add(correctness_disagreement))

        return stats

    def get_metadata(self) -> List[MetricMetadata]:
        """Return metadata for all computed metrics."""
        return [
            MetricMetadata(
                name="inter_model_agreement",
                display_name="Inter-Model Agreement",
                short_display_name="Agreement",
                description="Average pairwise agreement between models' outputs. Higher values indicate more consensus.",
                lower_is_better=False,
                group="cross_model",
            ),
            MetricMetadata(
                name="accuracy_consistency",
                display_name="Accuracy Consistency",
                short_display_name="Acc Consistency",
                description="Consistency of accuracy across models. Higher values indicate more consistent performance.",
                lower_is_better=False,
                group="cross_model",
            ),
            MetricMetadata(
                name="consensus_ratio",
                display_name="Consensus Ratio",
                short_display_name="Consensus",
                description="Fraction of outputs that appear in multiple models. Higher values indicate more consensus.",
                lower_is_better=False,
                group="cross_model",
            ),
            MetricMetadata(
                name="consensus_confidence",
                display_name="Consensus Confidence",
                short_display_name="Consensus Conf",
                description="Average log probability for consensus outputs. Higher values indicate higher confidence in consensus.",
                lower_is_better=False,
                group="cross_model",
            ),
            MetricMetadata(
                name="disagreement_rate",
                display_name="Disagreement Rate",
                short_display_name="Disagreement",
                description="Rate at which models produce different outputs. Lower values indicate more agreement.",
                lower_is_better=True,
                group="cross_model",
            ),
            MetricMetadata(
                name="correctness_disagreement",
                display_name="Correctness Disagreement",
                short_display_name="Correctness Disagree",
                description="Whether models disagree on correctness (1.0) or agree (0.0).",
                lower_is_better=True,
                group="cross_model",
            ),
        ]

