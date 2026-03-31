---
title: Perturbations
---

# Perturbations

MedHELM perturbations and data augmentations live in the Python package `helm.benchmark.augmentations` (same module layout as Stanford HELM, mirrored in this [MedHELM codebase](https://github.com/PacificAI/medhelm/tree/main/src/helm/benchmark/augmentations)). The reference below is generated from the source with [mkdocstrings](https://mkdocstrings.github.io/) (same filtering approach as the [CRFM HELM Perturbations](https://crfm-helm.readthedocs.io/en/latest/perturbations/) page).

::: helm.benchmark.augmentations
    options:
      filters:
        - "^(?!test_).+_perturbation$"
        - ".+Perturbation$"
      show_submodules: true
      show_root_heading: false
      show_root_toc_entry: false
      members_order: alphabetical
