[comment]: <> (When using the img tag, which allows us to specify size, src has to be a URL.)
<img src="https://github.com/stanford-crfm/helm/raw/main/src/helm/benchmark/static/images/helm-logo.png" alt=""  width="800"/>

Welcome!  This repository contains all the assets for [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), 
which includes the following features:

- Collection of datasets in a standard format (e.g., NaturalQuestions)
- Collection of models accessible via a unified API (e.g., GPT-3, MT-NLG, OPT, BLOOM)
- Collection of metrics beyond accuracy (efficiency, bias, toxicity, etc.)
- Collection of perturbations for evaluating robustness and fairness (e.g., typos, dialect)
- Modular framework for constructing prompts from datasets
- Proxy server for managing accounts and providing unified interface to access models

To read more:

- [Setup](docs/setup.md): how to run the code
- [Code](docs/code.md): how to contribute new scenarios or models
- [Running the proxy server](docs/proxy-server.md)
- [Running the benchmark](docs/benchmark.md)
- [Deployment](docs/deployment.md): for CRFM maintainers of the proxy server