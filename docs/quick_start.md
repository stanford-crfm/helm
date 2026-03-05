---
layout: default
title: Quick Start
---

# Quick Start

Install the package from PyPI:

```sh
pip install crfm-helm
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
