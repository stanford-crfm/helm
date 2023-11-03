<!--intro-start-->

# Holistic Evaluation of Language Models

[comment]: <> (When using the img tag, which allows us to specify size, src has to be a URL.)
<img src="https://github.com/stanford-crfm/helm/raw/main/src/helm/benchmark/static/images/helm-logo.png" alt=""  width="800"/>

Welcome! The **`crfm-helm`** Python package contains code used in the **Holistic Evaluation of Language Models** project ([paper](https://arxiv.org/abs/2211.09110), [website](https://crfm.stanford.edu/helm/latest/)) by [Stanford CRFM](https://crfm.stanford.edu/). This package includes the following features:

- Collection of datasets in a standard format (e.g., NaturalQuestions)
- Collection of models accessible via a unified API (e.g., GPT-3, MT-NLG, OPT, BLOOM)
- Collection of metrics beyond accuracy (efficiency, bias, toxicity, etc.)
- Collection of perturbations for evaluating robustness and fairness (e.g., typos, dialect)
- Modular framework for constructing prompts from datasets
- Proxy server for managing accounts and providing unified interface to access models
<!--intro-end-->

To get started, refer to [the documentation on Read the Docs](https://crfm-helm.readthedocs.io/) for how to install and run the package.

## Directory Structure

The directory structure for this repo is as follows

```
├── docs # MD used to generate readthedocs
│
├── scripts # Python utility scripts for HELM
│ ├── cache
│ ├── data_overlap # Calculate train test overlap
│ │ ├── common
│ │ ├── scenarios
│ │ └── test
│ ├── efficiency
│ ├── fact_completion
│ ├── offline_eval
│ └── scale
└── src
├── helm # Benchmarking Scripts for HELM
│ │
│ ├── benchmark # Main Python code for running HELM
│ │ │
│ │ └── static # Current JS (Jquery) code for rendering front-end
│ │ │
│ │ └── ...
│ │
│ ├── common # Additional Python code for running HELM
│ │
│ └── proxy # Python code for external web requests
│
└── helm-frontend # New React Front-end
```
