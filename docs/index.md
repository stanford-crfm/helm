---
layout: default
title: MedHELM
---

# Holistic Evaluation of Language Models (HELM)

[comment]: <> (When using the img tag, which allows us to specify size, src has to be a URL.)
<img src="https://github.com/stanford-crfm/helm/raw/v0.5.4/helm-frontend/src/assets/helm-logo.png" alt="HELM logo" width="480"/>

<a href="https://github.com/PacificAI/medhelm"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PacificAI/medhelm"></a>
<a href="https://github.com/PacificAI/medhelm/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/PacificAI/medhelm"></a>
<a href="https://github.com/PacificAI/medhelm/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/PacificAI/medhelm?color=blue" /></a>
<a href="https://pypi.org/project/medhelm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/medhelm?color=blue" /></a>

**Holistic Evaluation of Language Models (HELM)** is an open source Python framework created by the [Center for Research on Foundation Models (CRFM) at Stanford](https://crfm.stanford.edu/) for holistic, reproducible and transparent evaluation of foundation models, including large language models (LLMs) and multimodal models. This framework includes the following features:

- Datasets and benchmarks in a standardized format (e.g. MMLU-Pro, GPQA, IFEval, WildBench)
- Models from various providers accessible through a unified interface (e.g. OpenAI models, Anthropic Claude, Google Gemini)
- Metrics for measuring various aspects beyond accuracy (e.g. efficiency, bias, toxicity)
- Web UI for inspecting individual prompts and responses
- Web leaderboard for comparing results across models and benchmarks

## Documentation

Please refer to the documentation on this site (see **Docs**, **Tutorials**, **Benchmarks**) for instructions on how to install and run HELM.

## Quick Start

Install the package from PyPI:

```sh
pip install medhelm
```

Run the following in your shell:

```sh
# Run benchmark
helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite my-suite

# Start a web server to display benchmark results
helm-server --suite my-suite
```

Then go to <http://localhost:8000/> in your browser.

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
