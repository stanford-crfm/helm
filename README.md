# MedHELM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PacificAI/medhelm/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/medhelm?color=blue)](https://pypi.org/project/medhelm/)

<img src="https://github.com/PacificAI/medhelm/raw/main/docs/assets/images/medhelm_logo.jpg" alt="MedHELM" width="320"/>

## MedHELM

**MedHELM** is a multi-institutional effort to develop standardized, clinically grounded benchmarks for evaluating large language models in healthcare. While it builds on the HELM evaluation framework, MedHELM is independently developed through a broad collaboration spanning **Stanford Medicine**, **HAI**, **Microsoft**, and partners across the healthcare and research ecosystem.

The initiative focuses on real-world clinical tasks, emphasizing:
  - Transparency
  - Reproducibility
  - Practical relevance for healthcare deployment

This framework includes the following features:

  - Datasets and benchmarks in a standardized format (e.g. MMLU-Pro, GPQA, IFEval, WildBench)
  - Models from various providers accessible through a unified interface (e.g. OpenAI models, Anthropic Claude, Google Gemini)
  - Metrics for measuring various aspects beyond accuracy (e.g. efficiency, bias, toxicity)
  - Web UI for inspecting individual prompts and responses
  - Web leaderboard for comparing results across models and benchmarks

## Documentation

Documentation: **[medhelm.org](https://medhelm.org)**

## Install & run (MedHELM library)

MedHELM uses the HELM core engine and adds medical benchmarks.

### Using uv (recommended)

1. Create a virtual environment with Python 3.12:
```sh
uv venv --python 3.12 .venv
```

2. Activate the environment:
```sh
source .venv/bin/activate
```

3. Install MedHELM:

- If you cloned this repository (recommended for development / contributing):
```sh
uv pip install -e .
```

- If you want the released package from PyPI:

```sh
uv pip install medhelm
```

### Standard tier (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

**Quick test** (small model, 2 instances — runs in seconds):

```sh
medhelm-run --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite my_med_test --max-eval-instances 2
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

**Full example** (better quality, 10 instances):

```sh
medhelm-run --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite my_med_test --max-eval-instances 10
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

Then open http://localhost:8000/ in your browser.

### Alternative: Using pip

If you prefer `pip` instead of `uv`:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install medhelm
```

### Clinical NLP tier (`[summarization]`)

Adds heavy libraries (bert-score, rouge-score, nltk). **Install can take 2–3 minutes.**

Scenarios: **DischargeMe** (hospital course summaries; requires PhysioNet `data_path`), **ACI-Bench** (clinical transcripts), **Patient-Edu** (simplifying medical jargon).

**⚠️ Note on macOS with Python 3.12:** Use `pip` instead of `uv` for this tier due to build compatibility issues with `pyemd`.

Install (use `pip` on Intel Mac):
```sh
pip install "medhelm[summarization]"
```

Or with `uv` (Linux/other platforms):
```sh
uv pip install "medhelm[summarization]"
```

Example (ACI-Bench; runs without extra data):

```sh
medhelm-run --run-entries "aci_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite med_summaries --max-eval-instances 5
helm-summarize --suite med_summaries -o ./benchmark_output
helm-server --suite med_summaries -o ./benchmark_output --port 8000
```

### Gated / licensing tier (`[gated]`)

Adds **gdown** for scenarios that use Google Drive. Install can also take longer.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

Install:
```sh
uv pip install "medhelm[gated]"
```

Example:

```sh
medhelm-run --run-entries "med_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite board_exams --max-eval-instances 10
helm-summarize --suite board_exams -o ./benchmark_output
helm-server --suite board_exams -o ./benchmark_output --port 8000
```

### Classic HELM commands

You can still use `helm-run`, `helm-summarize`, and `helm-server`; `medhelm-run` is an alias for `helm-run`.

After activating your environment:

```sh
medhelm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10
helm-summarize --suite my-suite -o ./benchmark_output
helm-server --suite my-suite -o ./benchmark_output --port 8000
```

## Quick Start (summary)

<!--quick-start-begin-->

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `pip install medhelm` or `uv pip install medhelm` | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** (Clinical NLP tier) | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (2–3 min install; bert-score, rouge-score, nltk) |
| **Gated** (licensing tier) | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Drive; gdown) |

**Fast test:** `pubmed_qa` with `model=openai/gpt2,model_deployment=huggingface/gpt2` and `--max-eval-instances 2`. **Full run:** use `qwen/qwen2.5-7b-instruct` and more instances. Run `helm-summarize` and `helm-server` after. See [medhelm.org](https://medhelm.org) for full docs.

<!--quick-start-end-->

## Goals & roadmap

MedHELM aims to be a **new public repo** with **fewer dependencies**, **easier installation**, and **public documentation**. We welcome feedback on the following:

- **HealthBench:** We are considering new subcategories to include HealthBench. Do you see value in adding HealthBench, and how would you use it?
- **Non-gated alternatives:** We provide **7 non-gated datasets** (e.g. PubMedQA, MedCalc-Bench, MedicationQA, MedHallu, and others in the Standard and Summarization tiers) as free alternatives for the same kinds of tasks as gated benchmarks.
- **Hospital & private data:** We want to make it **easier for hospital systems to contribute or add their own private datasets**. If your institution is interested in running or contributing benchmarks, we’d like to hear from you.

## Leaderboard

We maintain a **medical** leaderboard for comparing models on MedHELM benchmarks:

- **[MedHELM Leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)** — PubMedQA, MedQA, MedMCQA, and other medical benchmarks.

To reproduce or extend results locally, see [medhelm.org](https://medhelm.org) (Reproducing Leaderboards, MedHELM docs).

## Citation

MedHELM builds on the Holistic Evaluation of Language Models framework. If you use this software in your research, please cite the MedHELM and HELM papers as below.

```bibtex

@Article{Bedi2026,
author={Bedi, Suhana and Cui, Hejie and Fuentes, Miguel and Unell, Alyssa and Wornow, Michael and Banda, Juan M. and Kotecha, Nikesh and Keyes, Timothy and Mai, Yifan and Oez, Mert and Qiu, Hao and Jain, Shrey and Schettini, Leonardo and Kashyap, Mehr and Fries, Jason Alan and Swaminathan, Akshay and Chung, Philip and Haredasht, Fateme Nateghi and Lopez, Ivan and Aali, Asad and Tse, Gabriel and Nayak, Ashwin and Vedak, Shivam and Jain, Sneha S. and Patel, Birju and Fayanju, Oluseyi and Shah, Shreya and Goh, Ethan and Yao, Dong-han and Soetikno, Brian and Reis, Eduardo and Gatidis, Sergios and Divi, Vasu and Capasso, Robson and Saralkar, Rachna and Chiang, Chia-Chun and Jindal, Jenelle and Pham, Tho and Ghoddusi, Faraz and Lin, Steven and Chiou, Albert S. and Hong, Hyo Jung and Roy, Mohana and Gensheimer, Michael F. and Patel, Hinesh and Schulman, Kevin and Dash, Dev and Char, Danton and Downing, Lance and Grolleau, Francois and Black, Kameron and Mieso, Bethel and Zahedivash, Aydin and Yim, Wen-wai and Sharma, Harshita and Lee, Tony and Kirsch, Hannah and Lee, Jennifer and Ambers, Nerissa and Lugtu, Carlene and Sharma, Aditya and Mawji, Bilal and Alekseyev, Alex and Zhou, Vicky and Kakkar, Vikas and Helzer, Jarrod and Revri, Anurang and Bannett, Yair and Daneshjou, Roxana and Chen, Jonathan and Alsentzer, Emily and Morse, Keith and Ravi, Nirmal and Aghaeepour, Nima and Kennedy, Vanessa and Chaudhari, Akshay and Wang, Thomas and Koyejo, Sanmi and Lungren, Matthew P. and Horvitz, Eric and Liang, Percy and Pfeffer, Michael A. and Shah, Nigam H.},
title={Holistic evaluation of large language models for medical tasks with MedHELM},
journal={Nature Medicine},
year={2026},
month={Mar},
day={01},
volume={32},
number={3},
pages={943-951},
abstract={While large language models (LLMs) achieve near-perfect scores on medical licensing exams, these evaluations inadequately reflect the complexity and diversity of real-world clinical practice. Here we introduce MedHELM, an extensible evaluation framework with three contributions. First, a clinician-validated taxonomy organizing medical AI applications into five categories that mirror real clinical tasks---clinical decision support (diagnostic decisions, treatment planning), clinical note generation (visit documentation, procedure reports), patient communication (education materials, care instructions), medical research (literature analysis, clinical data analysis) and administration (scheduling, workflow coordination). These encompass 22 subcategories and 121 specific tasks reflecting daily medical practice. Second, a comprehensive benchmark suite of 37 evaluations covering all subcategories. Third, systematic comparison of nine frontier LLMs---Claude 3.5 Sonnet, Claude 3.7 Sonnet, DeepSeek R1, Gemini 1.5 Pro, Gemini 2.0 Flash, GPT-4o, GPT-4o mini, Llama 3.3 and o3-mini---using an automated LLM-jury evaluation method. Our LLM-jury uses multiple AI evaluators to assess model outputs against expert-defined criteria. Advanced reasoning models (DeepSeek R1, o3-mini) demonstrated superior performance with win rates of 66{\%}, although Claude 3.5 Sonnet achieved comparable results at 15{\%} lower computational cost. These results not only highlight current model capabilities but also demonstrate how MedHELM could enable evidence-based selection of medical AI systems for healthcare applications.},
issn={1546-170X},
doi={10.1038/s41591-025-04151-2},
url={https://doi.org/10.1038/s41591-025-04151-2}
}
```
