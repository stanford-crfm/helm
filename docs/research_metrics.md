# Research-Level Metrics for HELM

This document describes the advanced research-level metrics available in HELM for uncertainty quantification, robustness analysis, and cross-model consistency evaluation.

## Overview

HELM now includes three comprehensive metric suites designed for research-level evaluation:

1. **Uncertainty Quantification Metrics**: Measure model confidence, prediction reliability, and uncertainty decomposition
2. **Robustness Metrics**: Evaluate model stability, sensitivity to perturbations, and adversarial robustness
3. **Cross-Model Consistency Metrics**: Analyze agreement and consensus between multiple models

These metrics are particularly valuable for:
- Understanding model calibration and confidence
- Evaluating robustness to adversarial inputs
- Analyzing ensemble potential and model consensus
- Research on trustworthy AI systems

## Uncertainty Quantification Metrics

### Overview

The `UncertaintyQuantificationMetric` provides comprehensive uncertainty analysis based on token-level log probabilities. It computes entropy-based measures, confidence intervals, and uncertainty decomposition.

### Key Metrics

- **Prediction Entropy**: Shannon entropy of the predicted probability distribution
- **Max Probability**: Maximum probability assigned to any outcome (confidence measure)
- **Entropy Ratio**: Normalized entropy (0 = certain, 1 = uniform)
- **Effective Outcomes**: Exponential of entropy (effective number of distinct outcomes)
- **Coverage Metrics**: Fraction of outcomes needed to achieve specific confidence levels (e.g., 80%, 90%, 95%)
- **Top-K Confidence**: Sum of probabilities for top K outcomes
- **Aleatoric/Epistemic Uncertainty**: Decomposition of uncertainty into data-driven vs model-driven components

### Usage

```python
from helm.benchmark.metrics.common_metric_specs import get_uncertainty_quantification_metric_specs

# Basic usage with default settings
metric_specs = get_uncertainty_quantification_metric_specs()

# Custom configuration
metric_specs = get_uncertainty_quantification_metric_specs(
    num_bins=20,
    confidence_levels=[0.8, 0.9, 0.95, 0.99],
    compute_uncertainty_decomposition=True
)
```

### Example Run Entry

```bash
helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt-4 \
  --suite uncertainty-analysis \
  --max-eval-instances 100
```

Then add the uncertainty metrics to your run spec configuration.

### Interpretation

- **High Entropy**: Model is uncertain (uniform distribution)
- **Low Entropy**: Model is confident (concentrated distribution)
- **High Aleatoric Ratio**: Uncertainty is primarily due to inherent data randomness
- **High Epistemic Uncertainty**: Uncertainty is due to model limitations

## Robustness Metrics

### Overview

The `RobustnessMetric` evaluates model stability, sensitivity to perturbations, and adversarial robustness. It compares outputs across multiple trials and perturbations.

### Key Metrics

- **Output Consistency**: Consistency of outputs across multiple trials
- **Logprob Stability**: Stability of log probabilities across trials
- **Perturbation Sensitivity**: Fraction of perturbations that change the output
- **Logprob Sensitivity**: Average change in log probability under perturbations
- **Adversarial Robustness**: Accuracy under worst-case perturbations
- **Robustness Drop**: Drop in accuracy from original to worst-case perturbations

### Usage

```python
from helm.benchmark.metrics.common_metric_specs import get_robustness_metric_specs

# Basic usage
metric_specs = get_robustness_metric_specs()

# Custom configuration
metric_specs = get_robustness_metric_specs(
    compute_sensitivity=True,
    compute_stability=True,
    compute_adversarial=True
)
```

### Requirements

- **Multiple Trials**: For stability metrics, set `num_train_trials > 1` in your adapter spec
- **Perturbations**: For sensitivity metrics, include perturbed instances in your scenario

### Example Run Entry

```bash
helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt-4 \
  --suite robustness-analysis \
  --max-eval-instances 100 \
  --num-train-trials 5
```

### Interpretation

- **High Output Consistency**: Model produces stable predictions across trials
- **Low Perturbation Sensitivity**: Model is robust to input variations
- **High Adversarial Robustness**: Model maintains accuracy under adversarial inputs
- **Low Robustness Drop**: Model performance doesn't degrade significantly under perturbations

## Cross-Model Consistency Metrics

### Overview

The `CrossModelConsistencyMetric` analyzes agreement and consensus between multiple models evaluated on the same instances. This is useful for ensemble analysis and understanding model disagreement patterns.

### Key Metrics

- **Inter-Model Agreement**: Average pairwise agreement between models' outputs
- **Accuracy Consistency**: Consistency of accuracy across models
- **Consensus Ratio**: Fraction of outputs that appear in multiple models
- **Consensus Confidence**: Average log probability for consensus outputs
- **Disagreement Rate**: Rate at which models produce different outputs
- **Correctness Disagreement**: Whether models agree on correctness

### Usage

```python
from helm.benchmark.metrics.common_metric_specs import get_cross_model_consistency_metric_specs

# Basic usage
metric_specs = get_cross_model_consistency_metric_specs()

# Custom configuration
metric_specs = get_cross_model_consistency_metric_specs(
    compute_agreement=True,
    compute_consensus=True,
    compute_disagreement=True
)
```

### Requirements

- **Multiple Models**: Requires evaluating at least 2 models on the same instances
- **Same Instances**: Models must be evaluated on identical instances for meaningful comparison

### Example Run Entry

```bash
# Evaluate multiple models
helm-run --run-entries \
  mmlu:subject=philosophy,model=openai/gpt-4 \
  mmlu:subject=philosophy,model=anthropic/claude-3 \
  mmlu:subject=philosophy,model=google/gemini-pro \
  --suite cross-model-analysis \
  --max-eval-instances 100
```

### Interpretation

- **High Inter-Model Agreement**: Models produce similar outputs (good for ensembles)
- **High Consensus Ratio**: Many outputs appear in multiple models (strong consensus)
- **Low Disagreement Rate**: Models rarely produce different outputs
- **High Consensus Confidence**: Models are confident when they agree

## Using All Research Metrics Together

For comprehensive research evaluation, you can use all three metric suites together:

```python
from helm.benchmark.metrics.common_metric_specs import get_research_metrics_specs

# Get all research metrics
metric_specs = get_research_metrics_specs()
```

This includes:
- Uncertainty quantification metrics
- Robustness metrics
- Cross-model consistency metrics

## Integration with Existing Metrics

These research metrics are designed to complement existing HELM metrics:

- **Basic Metrics**: Accuracy, F1, etc. (what the model predicts)
- **Calibration Metrics**: ECE, Platt scaling (how well-calibrated predictions are)
- **Uncertainty Metrics**: Entropy, confidence intervals (how certain the model is)
- **Robustness Metrics**: Stability, sensitivity (how robust the model is)
- **Cross-Model Metrics**: Agreement, consensus (how models compare)

## Research Applications

### Model Calibration Research

Use uncertainty quantification metrics to:
- Study the relationship between confidence and accuracy
- Evaluate calibration methods
- Analyze uncertainty decomposition

### Robustness Research

Use robustness metrics to:
- Evaluate adversarial robustness
- Study sensitivity to perturbations
- Analyze stability across trials

### Ensemble Research

Use cross-model consistency metrics to:
- Identify ensemble potential
- Study model disagreement patterns
- Analyze consensus-based confidence

## Best Practices

1. **Sufficient Data**: Ensure you have enough instances for reliable statistics (recommended: 100+ instances)

2. **Multiple Trials**: For robustness metrics, use `num_train_trials > 1` to measure stability

3. **Perturbations**: Include perturbed instances to evaluate sensitivity

4. **Multiple Models**: For cross-model metrics, evaluate at least 2-3 models

5. **Interpretation**: Consider metrics together rather than in isolation

## Limitations

- **Uncertainty Metrics**: Require token-level log probabilities (not available for all models)
- **Robustness Metrics**: Require multiple trials or perturbations for meaningful results
- **Cross-Model Metrics**: Require evaluating multiple models on the same instances

## Future Enhancements

Potential future additions:
- Bayesian uncertainty quantification
- More sophisticated uncertainty decomposition methods
- Additional robustness metrics (e.g., certified robustness)
- Ensemble performance prediction

## Citation

If you use these metrics in your research, please cite:

```bibtex
@article{liang2023holistic,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

## Contributing

We welcome contributions to expand these research metrics. Please see the [Contributing Guide](../CONTRIBUTING.md) for details.

