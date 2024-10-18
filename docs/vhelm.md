# VHELM (Vision-Language Models)

**Holistic Evaluation of Vision-Language Models (VHELM)** is an extension of the HELM framework for evaluating **Vision-Language Models (VLMs)**.

VHELM aggregates various datasets to cover one or more of the 9 aspects: visual perception, bias, fairness, knowledge, multilinguality, reasoning, robustness, safety, and toxicity. In doing so, we produce a comprehensive, multi-dimensional view of the capabilities of the VLMs across these important factors. In addition, we standardize the standard inference parameters, methods of prompting, and evaluation metrics to enable fair comparisons across models.

## References

- [Leaderboard](https://crfm.stanford.edu/helm/vhelm/v2.0.1/)
- Paper (TBD)

## Installation

First, follow the [installation instructions](installation.md) to install the base HELM Python page.

To install the additional dependencies to run VHELM, run:

```sh
pip install "crfm-helm[vlm]"
```

## Quick Start

The following is an example of evaluating `openai/gpt-4o-mini-2024-07-18` on 10 instance from the Accounting subset of MMMU.

```sh
# Download schema_vhelm.yaml
wget https://raw.githubusercontent.com/stanford-crfm/helm/refs/heads/main/src/helm/benchmark/static/schema_vhelm.yaml

# Run benchmark
helm-run --run-entries mmmu:subject=Accounting,model=openai/gpt-4o-mini-2024-07-18 --suite my-vhelm-suite --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite my-vhelm-suite --schema-path schema_vhelm.yaml

# Start a web server to display benchmark results
helm-server --suite my-vhelm-suite
```

Then go to http://localhost:8000/ in your browser.

## Reproducing the Leaderboard

To reproduce the [entire VHELM leaderboard](https://crfm.stanford.edu/helm/vhelm/latest/), refer to the instructions for VHELM on the [Reproducing Leaderboards](reproducing_leaderboards.md) documentation.
