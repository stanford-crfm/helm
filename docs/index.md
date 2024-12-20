{%
   include-markdown "../README.md"
   start="<!--intro-start-->"
   end="<!--intro-end-->"
%}

The code is [hosted on GitHub here](https://github.com/stanford-crfm/helm/).

To run the code, refer to the User Guide's chapters:

- [Installation](installation.md)
- [Quick Start](quick_start.md)
- [Tutorial](tutorial.md)
- [Get Your Model's Leaderboard Rank](get_helm_rank.md)

To add new models and scenarios, refer to the Developer Guide's chapters:

- [Developer Setup](developer_setup.md)
- [Code Structure](code.md)

## Papers

This repository contains code used to produce results for the following papers:

- **Holistic Evaluation of Vision-Language Models (VHELM)** - [paper](https://arxiv.org/abs/2410.07112), [leaderboard](https://crfm.stanford.edu/helm/vhelm/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/vhelm/)
- **Holistic Evaluation of Text-To-Image Models (HEIM)** - [paper](https://arxiv.org/abs/2311.04287), [leaderboard](https://crfm.stanford.edu/helm/heim/latest/), [documentation](https://crfm-helm.readthedocs.io/en/latest/heim/)
- **Enterprise Benchmarks for Large Language Model Evaluation** - [paper](https://arxiv.org/abs/2410.12857), [documentation](https://crfm-helm.readthedocs.io/en/latest/enterprise_benchmark/)

The HELM Python package can be used to reproduce the published model evaluation results from these papers. To get started, refer to the documentation links above for the corresponding paper, or the [main Reproducing Leaderboards documentation](https://crfm-helm.readthedocs.io/en/latest/reproducing_leaderboards/).