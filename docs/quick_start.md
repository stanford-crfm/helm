---
layout: default
title: Quick Start
---

# Quick Start

MedHELM is a public Python library with fewer dependencies and straightforward installation. Install from PyPI and run benchmarks with the **uv** workflow (or use `pip` and `helm-run` if you prefer).

## Standard (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

```sh
uv pip install medhelm
```

Run a benchmark:

```sh
uv run medhelm-run \
  --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite my_med_test \
  --max-eval-instances 10
uv run helm-summarize --suite my_med_test
uv run helm-server --suite my_med_test
```

Then open <http://localhost:8000/> in your browser.

## Clinical NLP tier (summarization)

Adds heavier libraries (bert-score, rouge-score, nltk). **Install may take 2–3 minutes.**

Scenarios: **DischargeMe** (hospital course summaries; requires PhysioNet `data_path`), **ACI-Bench** (clinical transcripts), **Patient-Edu** (simplifying medical jargon).

```sh
uv pip install "medhelm[summarization]"
```

Example (ACI-Bench; runs without extra data):

```sh
uv run medhelm-run \
  --run-entries "aci_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite med_summaries \
  --max-eval-instances 5
uv run helm-summarize --suite med_summaries
uv run helm-server --suite med_summaries
```

## Gated / licensing tier (Drive scenarios)

Adds **gdown** so the code can download data from Google Drive. Install can also take longer.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

```sh
uv pip install "medhelm[gated]"
```

Example:

```sh
uv run medhelm-run \
  --run-entries "med_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite board_exams \
  --max-eval-instances 10
uv run helm-summarize --suite board_exams
uv run helm-server --suite board_exams
```

## Summary

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `uv pip install medhelm` | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `uv pip install "medhelm[summarization]"` | DischargeMe (needs data_path), ACI-Bench, Patient-Edu (2–3 min install) |
| **Gated** | `uv pip install "medhelm[gated]"` | MedQA, MedMCQA (Drive) |

You can use `pip install medhelm` (and `pip install "medhelm[summarization]"` / `pip install "medhelm[gated]"`) instead of `uv pip install`; then run with `medhelm-run` (or `helm-run`), `helm-summarize`, and `helm-server`.
