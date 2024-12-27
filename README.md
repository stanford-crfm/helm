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

## Papers

This repository contains code used to produce results for the following papers:

- **Holistic Evaluation of Vision-Language Models (VHELM)** - [paper](https://arxiv.org/abs/2410.07112), [leaderboard](https://crfm.stanford.edu/helm/vhelm/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/vhelm/)
- **Holistic Evaluation of Text-To-Image Models (HEIM)** - [paper](https://arxiv.org/abs/2311.04287), [leaderboard](https://crfm.stanford.edu/helm/heim/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/heim/)
- **Enterprise Benchmarks for Large Language Model Evaluation** - [paper](https://arxiv.org/abs/2410.12857), [documentation](https://crfm-helm.readthedocs.io/en/latest/enterprise_benchmark/)

The HELM Python package can be used to reproduce the published model evaluation results from these papers. To get started, refer to the documentation links above for the corresponding paper, or the [main Reproducing Leaderboards documentation](https://crfm-helm.readthedocs.io/en/latest/reproducing_leaderboards/).

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