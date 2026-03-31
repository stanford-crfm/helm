---
title: Metrics
---

# Metrics

MedHELM evaluation metrics are implemented in the Python package `helm.benchmark.metrics` (same module layout as Stanford HELM, mirrored in this [MedHELM codebase](https://github.com/PacificAI/medhelm/tree/main/src/helm/benchmark/metrics)). The reference below is generated from the source with [mkdocstrings](https://mkdocstrings.github.io/) (same filtering approach as the [CRFM HELM Metrics](https://crfm-helm.readthedocs.io/en/latest/metrics/) page).

::: helm.benchmark.metrics
    options:
      filters:
        - "^(?!test_).+_metrics$"
        - "Metric$"
        - "^evaluate_"
      show_submodules: true
      show_root_heading: false
      show_root_toc_entry: false
      members_order: alphabetical
