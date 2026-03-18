---
layout: default
title: MedHELM
---

<div class="home-logos">
  <a href="https://crfm.stanford.edu/" rel="noopener" target="_blank" aria-label="Stanford CRFM"><img src="{{ '/assets/images/crfm-logo.png' | relative_url }}" alt="Stanford CRFM" height="48"/></a>
  <a href="https://crfm.stanford.edu/helm/" rel="noopener" target="_blank" aria-label="HELM"><img src="{{ '/assets/images/helm-logo.png' | relative_url }}" alt="HELM" height="48"/></a>
  <a href="https://hai.stanford.edu/" rel="noopener" target="_blank" aria-label="Stanford HAI"><img src="{{ '/assets/images/chai-logo.svg' | relative_url }}" alt="Stanford HAI" height="48"/></a>
  <a href="https://pacific.ai/" rel="noopener" target="_blank" aria-label="Pacific AI"><img src="{{ '/assets/images/pacific-ai-logo.svg' | relative_url }}" alt="Pacific AI" height="48"/></a>
</div>

# MedHELM

<img src="{{ '/assets/images/medhelm_logo.jpg' | relative_url }}" alt="MedHELM" width="160"/>

<a href="https://github.com/PacificAI/medhelm/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/PacificAI/medhelm"></a>
<a href="https://github.com/PacificAI/medhelm/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/medhelm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/medhelm?color=blue" /></a>

**License:** This project is licensed under the [Apache License 2.0](https://github.com/PacificAI/medhelm/blob/main/LICENSE).

**Holistic Evaluation of Language Models (HELM)** is an open source Python framework created by the [Center for Research on Foundation Models (CRFM) at Stanford](https://crfm.stanford.edu/) for holistic, reproducible and transparent evaluation of foundation models, including large language models (LLMs) and multimodal models. This framework includes the following features:

- Datasets and benchmarks in a standardized format (e.g. MMLU-Pro, GPQA, IFEval, WildBench)
- Models from various providers accessible through a unified interface (e.g. OpenAI models, Anthropic Claude, Google Gemini)
- Metrics for measuring various aspects beyond accuracy (e.g. efficiency, bias, toxicity)
- Web UI for inspecting individual prompts and responses
- Web leaderboard for comparing results across models and benchmarks

## Documentation

Please refer to the documentation on this site (see **Docs**, **Tutorials**, **Benchmarks**) for instructions on how to install and run HELM.

## Quick Start

MedHELM is a public library with fewer dependencies and straightforward installation. Install from PyPI:

**Standard** (PubMedQA, MedCalc-Bench, MedicationQA, MedHallu):

```sh
uv pip install medhelm
```

Run a benchmark:

```sh
uv run medhelm-run --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite my_med_test --max-eval-instances 10
uv run helm-summarize --suite my_med_test
uv run helm-server --suite my_med_test
```

Then go to <http://localhost:8000/> in your browser.

- **Summarization tier** (DischargeMe, ACI-Bench, Patient-Edu): `uv pip install "medhelm[summarization]"` — example: `aci_bench:model=...,model_deployment=...` (install may take 2–3 min).
- **Gated tier** (MedQA, MedMCQA, Drive): `uv pip install "medhelm[gated]"`.

See [Quick Start](quick_start.html) and [Installation](installation.html) for full details.

## Goals & roadmap

MedHELM aims to be a **new public repo** with **fewer dependencies**, **easier installation**, and **public documentation**. We welcome feedback on:

- **HealthBench:** We are considering new subcategories to include HealthBench. Do you see value in adding HealthBench, and how would you use it?
- **Non-gated alternatives:** We provide **7 non-gated datasets** (e.g. PubMedQA, MedCalc-Bench, MedicationQA, MedHallu, and others in the Standard and Summarization tiers) as free alternatives for the same kinds of tasks as gated benchmarks.
- **Hospital & private data:** We want to make it **easier for hospital systems to contribute or add their own private datasets**. If your institution is interested in running or contributing benchmarks, we’d like to hear from you.

## Leaderboards

We maintain official leaderboards with results from evaluating recent models on notable benchmarks using this framework. Our current flagship leaderboards are:

- [HELM Capabilities](https://crfm.stanford.edu/helm/capabilities/latest/)
- [HELM Safety](https://crfm.stanford.edu/helm/safety/latest/)
- [Holistic Evaluation of Vision-Language Models (VHELM)](https://crfm.stanford.edu/helm/vhelm/latest/)

We also maintain leaderboards for a diverse range of domains (e.g. medicine, finance) and aspects (e.g. multi-linguality, world knowledge, regulation compliance). Refer to the [HELM website](https://crfm.stanford.edu/helm/) for a full list of leaderboards.

## Papers

The HELM framework was used in the following papers for evaluating models.

- **Holistic Evaluation of Language Models** - [paper](https://openreview.net/forum?id=iO4LZibEqW), [leaderboard](https://crfm.stanford.edu/helm/classic/latest/)
- **Holistic Evaluation of Vision-Language Models (VHELM)** - [paper](https://arxiv.org/abs/2410.07112), [leaderboard](https://crfm.stanford.edu/helm/vhelm/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/vhelm/)
- **Holistic Evaluation of Text-To-Image Models (HEIM)** - [paper](https://arxiv.org/abs/2311.04287), [leaderboard](https://crfm.stanford.edu/helm/heim/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/heim/)
- **Image2Struct: Benchmarking Structure Extraction for Vision-Language Models** - [paper](https://arxiv.org/abs/2410.22456)
- **Enterprise Benchmarks for Large Language Model Evaluation** - [paper](https://arxiv.org/abs/2410.12857), [documentation](https://crfm-helm.readthedocs.io/en/latest/enterprise_benchmark/)
- **The Mighty ToRR: A Benchmark for Table Reasoning and Robustness** - [paper](https://arxiv.org/abs/2502.19412), [leaderboard](https://crfm.stanford.edu/helm/torr/latest/)
- **Reliable and Efficient Amortized Model-based Evaluation** - [paper](https://arxiv.org/abs/2503.13335), [documentation](https://crfm-helm.readthedocs.io/en/latest/reeval/)
- **MedHELM** - paper in progress, [leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/), [documentation](/medhelm/)
- **Holistic Evaluation of Audio-Language Models** - [paper](https://arxiv.org/abs/2508.21376), [leaderboard](https://crfm.stanford.edu/helm/audio/latest/)

The HELM framework can be used to reproduce the published model evaluation results from these papers. To get started, refer to the documentation links above for the corresponding paper, or the [Reproducing Leaderboards](/reproducing_leaderboards/) documentation.

## Citation

If you use this software in your research, please cite the [Holistic Evaluation of Language Models paper](https://openreview.net/forum?id=iO4LZibEqW) as below.

```bibtex
@article{
liang2023holistic,
title={Holistic Evaluation of Language Models},
author={Percy Liang and Rishi Bommasani and Tony Lee and Dimitris Tsipras and Dilara Soylu and Michihiro Yasunaga and Yian Zhang and Deepak Narayanan and Yuhuai Wu and Ananya Kumar and Benjamin Newman and Binhang Yuan and Bobby Yan and Ce Zhang and Christian Alexander Cosgrove and Christopher D Manning and Christopher Re and Diana Acosta-Navas and Drew Arad Hudson and Eric Zelikman and Esin Durmus and Faisal Ladhak and Frieda Rong and Hongyu Ren and Huaxiu Yao and Jue WANG and Keshav Santhanam and Laurel Orr and Lucia Zheng and Mert Yuksekgonul and Mirac Suzgun and Nathan Kim and Neel Guha and Niladri S. Chatterji and Omar Khattab and Peter Henderson and Qian Huang and Ryan Andrew Chi and Sang Michael Xie and Shibani Santurkar and Surya Ganguli and Tatsunori Hashimoto and Thomas Icard and Tianyi Zhang and Vishrav Chaudhary and William Wang and Xuechen Li and Yifan Mai and Yuhui Zhang and Yuta Koreeda},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=iO4LZibEqW},
note={Featured Certification, Expert Certification}
}
```
