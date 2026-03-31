---
title: Scenarios
---

# Scenarios

MedHELM scenarios are implemented in the Python package `helm.benchmark.scenarios` (same module layout as Stanford HELM, mirrored in this [MedHELM codebase](https://github.com/PacificAI/medhelm/tree/main/src/helm/benchmark/scenarios)). The reference below is generated from the source with [mkdocstrings](https://mkdocstrings.github.io/) (same filtering approach as the [CRFM HELM Scenarios](https://crfm-helm.readthedocs.io/en/latest/scenarios/) page).

::: helm.benchmark.scenarios
    options:
      filters:
        - "^(?!test_).+_scenario$"
        - "Scenario$"
      show_submodules: true
      show_root_heading: false
      show_root_toc_entry: false
      members_order: alphabetical
