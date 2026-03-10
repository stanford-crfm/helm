---
title: Installation
---
# Installation

## Create a virtual environment

It is recommended to install MedHELM into a virtual environment with Python version >=3.10 to avoid dependency conflicts. MedHELM requires Python >=3.10. To create and activate a Python virtual environment, follow the instructions below.

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.10 helm-venv

# Activate the virtual environment.
source helm-venv/bin/activate
```

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n medhelm python=3.10 pip

# Activate the virtual environment.
conda activate medhelm
```

## Install MedHELM

Choose one of the following tiers depending on which scenarios you want to run.

### Standard (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

```
pip install medhelm
```

With [uv](https://docs.astral.sh/uv/):

```
uv pip install medhelm
```

### Clinical NLP tier (`[summarization]`)

Adds heavier libraries (bert-score, rouge-score, nltk). **Install may take 2–3 minutes.**

Scenarios: **DischargeMe** (hospital course summaries), **ACI-Bench** (clinical transcripts), **Patient-Edu** (simplifying medical jargon).

```
pip install "medhelm[summarization]"
```

With uv:

```
uv pip install "medhelm[summarization]"
```

### Gated / licensing tier (`[gated]`)

Adds **gdown** so the code can download data from Google Drive. Install can also take longer.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

```
pip install "medhelm[gated]"
```

With uv:

```
uv pip install "medhelm[gated]"
```

## Summary

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `pip install medhelm` or `uv pip install medhelm` | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (2–3 min install) |
| **Gated** | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive) |

See [Quick Start](quick_start.html) for running benchmarks with `uv run medhelm-run`.

## Install Multimodal Support

Additional steps are required for multimodal evaluations:

- **HEIM (Text-to-image Model Evaluation)** - to install the additional dependencies to run HEIM (text-to-image evaluation), refer to the [HEIM documentation](heim/).
- **VHELM (Vision-Language Models)** - To install the additional dependencies to run VHELM (Vision-Language Models), refer to the [VHELM documentation](vhelm/).
