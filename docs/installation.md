---
title: Installation
---
# Installation

## Create a virtual environment

MedHELM is compatible with Python 3.10, 3.11, and 3.12. We recommend **Python 3.12** for best compatibility.

It is recommended to install MedHELM into a virtual environment to avoid dependency conflicts.

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. To create and activate a Python virtual environment:

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12 .venv

# Activate the virtual environment
source .venv/bin/activate
```

### Using venv (Alternative)

```bash
# Create a virtual environment
python3.12 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

## Install MedHELM

Choose one of the following tiers depending on which scenarios you want to run. After activating your environment, use `pip` for installation (it handles legacy packages better than uv).

### Install from this repository (recommended if you cloned the repo)

If you cloned the MedHELM repository and want to run using the checked-out source:

```bash
uv pip install -e .
```

### Standard (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

```bash
pip install medhelm
```

### Clinical NLP tier (`[summarization]`)

Adds heavier libraries (bert-score, rouge-score, nltk). **Install may take 2–3 minutes.**

Scenarios: **DischargeMe** (hospital course summaries), **ACI-Bench** (clinical transcripts), **Patient-Edu** (simplifying medical jargon).

```bash
pip install "medhelm[summarization]"
```

**Note on macOS with Python 3.12:** Use `pip` (not `uv`) for this tier due to build compatibility with `pyemd`.

### Gated / licensing tier (`[gated]`)

Adds **gdown** so the code can download data from Google Drive. Install can also take longer.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

```bash
pip install "medhelm[gated]"
```

### Install all tiers at once

To install all scenarios (standard, summarization, and gated):

```bash
pip install "medhelm[summarization,gated]"
```

## Summary

| Tier | Install | Scenarios |
|------|--------|------------|
| **Standard** | `pip install medhelm` | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (2–3 min install) |
| **Gated** | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive) |
| **All tiers** | `pip install "medhelm[summarization,gated]"` | All of the above (install once, run any scenario) |

See [Quick Start](/quick_start) for running benchmarks with `medhelm-run` (after activating your environment).

## Troubleshooting

### macOS with Python 3.12 and the summarization tier

If you encounter build errors with `pyemd` when installing `medhelm[summarization]` on macOS:

```bash
pip install "medhelm[summarization]"
```

This is a known compatibility issue. Use `pip` instead of `uv` for this tier.

## Install Multimodal Support

Additional steps are required for multimodal evaluations:

- **HEIM (Text-to-image Model Evaluation)** - to install the additional dependencies to run HEIM (text-to-image evaluation), refer to the [HEIM documentation](/heim).
- **VHELM (Vision-Language Models)** - To install the additional dependencies to run VHELM (Vision-Language Models), refer to the [VHELM documentation](/vhelm).
