# MedHELM: Holistic Evaluation of Large Language Models for Medical Applications

> **Who it’s for:** Data scientists at health systems benchmarking LLMs for clinical and operations use cases.
>
> **What you’ll do here:** Install MedHELM, run a small evaluation locally, understand access levels, view leaderboards, and learn how to contribute new scenarios/models.
>
> **Time required:** \~10 minutes for the Quickstart.

MedHELM extends the HELM framework to evaluate **large language models (LLMs) in medical applications**, focusing on realistic tasks, safety, and reproducibility.


## Quickstart (10 minutes)

**Goal:** Install MedHELM, run a 10-instance evaluation on a public scenario, and open a local leaderboard.

**Prerequisites:** Python 3.10, ability to create a virtual environment via [conda](https://www.anaconda.com/docs/getting-started/getting-started).

### 1) Install (Python 3.10 recommended)

Run the following commands to create and activate a new python virtual environment:

```bash
# Create and activate a clean environment
conda create -n crfm-helm python=3.10 pip
conda activate crfm-helm
```

Run the following commands to install HELM and the necessary MedHELM extensions:

```
# Install HELM + MedHELM extras
pip install "crfm-helm[summarization]" && pip install "crfm-helm[medhelm]"
```

### 2) Run a tiny evaluation

```bash
helm-run \
  --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct \
  --suite my-medhelm-suite \
  --max-eval-instances 10
```

*Flags explained once:*

* `--run-entries`: which `(model, scenario)` to run
* `--suite`: groups runs and names the output folder slice
* `--max-eval-instances`: run a small subset for quick checks

### 3) Build and open the local leaderboard

```bash
helm-summarize --suite my-medhelm-suite --schema schema_medhelm.yaml
helm-server --suite my-medhelm-suite
```

**Expected outcome:**

* A local URL for the leaderboard (printed in the terminal)
* Benchmark results at `./benchmark_output/runs/my-medhelm-suite/`

> ⚠️ **PHI & compliance:** Only run **gated** or **private** data on infrastructure approved by your organization. Use the **redaction** steps in *Sharing Results* before sending outputs externally.


## Core Concepts

* **Scenario:** Dataset + prompt/response formatting logic.
* **Run entry:** A `(model, scenario)` pair given to `helm-run`.
* **Suite:** Named collection of runs; appears as a tab/section in the leaderboard.
* **Annotator:** Optional post‑processing (e.g., LLM‑as‑a‑judge).
* **Schema:** Task taxonomy + metrics configuration powering `helm-summarize` and the UI.
* **Release:** The version of leaderboard results.


## Clinician‑Validated Taxonomy (overview)

MedHELM evaluates models across a clinician‑validated taxonomy: **5 categories**, **22 subcategories**, **121 tasks**.

* **Clinical Decision Support**

  * Supporting Diagnostic Decisions
  * Planning Treatments
  * Predicting Patient Risks and Outcomes
  * Providing Clinical Knowledge Support

* **Clinical Note Generation**

  * Documenting Patient Visits
  * Recording Procedures
  * Documenting Diagnostic Reports
  * Documenting Care Plans

* **Patient Communication and Education**

  * Providing Patient Education Resources
  * Delivering Personalized Care Instructions
  * Patient‑Provider Messaging
  * Enhancing Patient Understanding and Accessibility in Health Communication
  * Facilitating Patient Engagement and Support

* **Medical Research Assistance**

  * Conducting Literature Research
  * Analyzing Clinical Research Data
  * Recording Research Processes
  * Ensuring Clinical Research Quality
  * Managing Research Enrollment

* **Administration and Workflow**

  * Scheduling Resources and Staff
  * Overseeing Financial Activities
  * Organizing Workflow Processes
  * Care Coordination and Planning


## Installation

**Prerequisites:** Python 3.10, ability to create a virtual environment (conda or venv). If using private/gated datasets, ensure your credentials (e.g., PhysioNet/Redivis) are approved by your org.

1. Follow the [installation instructions](installation.md) to install the base HELM framework. Do not install the multimodal support listed there.


2. Install the MedHELM-specific dependencies on the same python virtual environment from step 1 by running the following command:
    ```bash
    pip install "crfm-helm[summarization]" && pip install "crfm-helm[medhelm]"
    ```


## Run Your First Evaluation

The example below evaluates **Qwen2.5‑7B‑Instruct** on the **MedCalc‑Bench** scenario using 10 instances.


1. **Run the benchmark:** The following command runs **MedCalc-Bench** on **Qwen2.5‑7B‑Instruct** for 10 instances and stores the results under `./benchmark_output/runs/my-medhelm-suite`. 

   ```bash
   helm-run \
    --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct \
    --suite my-medhelm-suite \
    --max-eval-instances 10
   ```
2. **Create the leaderboard:** The following command converts the results from step 1 into an interactive leaderboard.
   ```bash
   helm-summarize --suite my-medhelm-suite --schema schema_medhelm.yaml
   ```
3. **Run the leaderboard locally:** This command runs the leaderboard on a local server. The exact address and port will show on the command output.

   ```bash
   helm-server --suite my-medhelm-suite
   ```


## Benchmark Access Levels

MedHELM scenarios fall into three access patterns. Use the right **run entries** file to register new runs and to reproduce results.

### Quick decision guide

* Dataset **fully public** → use `run_entries_medhelm_public.conf`
* Dataset **gated/public with credential** (e.g., PhysioNet) → `run_entries_medhelm_gated.conf`
* Dataset **private/organization‑only** → `run_entries_medhelm_private_{organization}.conf`

### Summary table

| Access type | Example sources                | Run entries file                         | Who can reproduce      |
| ----------- | ------------------------------ | ---------------------------------------- | ---------------------- |
| Public      | Hugging Face, GitHub           | `run_entries_medhelm_public.conf`        | Anyone                 |
| Gated       | PhysioNet, Redivis             | `run_entries_medhelm_gated.conf`         | Credentialed users     |
| Private     | Org‑internal clinical datasets | `run_entries_medhelm_private_{org}.conf` | Authorized org members |

> When contributing or reproducing results, ensure you’re using the correct file for the benchmark’s access level.
>
> For **private** benchmarks and organization‑specific configurations, contact the MedHELM team at **[migufuen@stanford.edu](mailto:migufuen@stanford.edu)**.


## Viewing and Reproducing Leaderboard Results

You can interact with MedHELM results by **viewing** pre‑computed results locally or by **reproducing** evaluations from scratch.

### View the official leaderboard locally

> **Prerequisite:** Complete Step 1 of the installation process in the [Installation](#installation) section.


1. **Download raw results** 
   
   Follow the instructions under [Downloading Raw Results](downloading_raw_results.md) up until the *Download a whole project* section to download the MedHELM leaderboard.
2. **Launch the local leaderboard:**
   
   Run the following command to launch the MedHELM leaderboard locally. Use the numbered `release` version you want to display. Check out all release versions on the upper right corner of the official [leaderboard website](https://crfm.stanford.edu/helm/medhelm/latest).

   ```bash
   # Sample command to launch the MedHELM leaderboard version 2.0.0.
   helm-server --release v2.0.0
   ```

### Reproduce leaderboard results

> **Prerequisite:** Complete the whole installation process in the [Installation](#installation) section.

* **Public benchmarks:** Anyone can reproduce the **public subset** using entries in `run_entries_medhelm_public.conf`.
* **Gated benchmarks:** Require credentials/approval (e.g., EHRSHOT); entries live in `run_entries_medhelm_gated.conf`.
* **Private benchmarks:** Org‑specific; entries follow `run_entries_medhelm_private_{organization}.conf`.

> **Note:** The `model_deployments` of the models listed in these run entries are specific to Stanford Healthcare, please change them for the appropriate deployments as needed. For more information on model_deployments, refer to [Adding New Models](adding_new_models.md).


## Contributing to MedHELM

We welcome contributions from research and clinical teams.

### Add a new scenario

1. **Create a scenario class** under `src/helm/benchmark/scenarios/` that transforms your dataset to HELM’s input format. Include:

   * Prompt (with any patient context if present)
   * Gold responses (if available)
   * Useful metadata for post‑processing

   See [Adding New Scenarios](adding_new_scenarios.md) for details.

2. **Define a run spec** in `src/helm/benchmark/run_specs/medhelm_run_specs.py` linking your scenario to the CLI. Specify:

   * Instructioning for models
   * Metrics to compute
   * Optional annotation steps

   *Examples:*

   * **Annotator:** `src/helm/benchmark/annotation/med_dialog_annotator.py`
   * **Custom metrics:** add under `src/helm/benchmark/metrics/` (e.g., `medcalc_bench_metrics.py`)

3. **Register run entries** for the leaderboard using the correct config (public/gated/private).

4. **Register your scenario in the schema** by adding it to `schema_medhelm.yaml` (task taxonomy + any new metrics).

### Add a new model

Follow [Adding New Models](adding_new_models.md) in your docs to register a model and its deployment (e.g., API, etc.).


## Sharing Results

Before sharing outputs generated from **gated** or **private** datasets:

1. **Redact prompts/responses** using `scripts/redact_scenario_states.py`:

   ```bash
   python3 scripts/redact_scenario_states.py \
     --output benchmark_output \
     --suite my_suite \
     --redact-output
   ```
2. **Redact annotations** if your scenario stores sensitive information under annotations (e.g., LLM‑as‑a‑judge outputs) using `scripts/medhelm/redact_annotations.py`:
   ```bash
   python3 scripts/medhelm/redact_annotations.py \
     --output benchmark_output \
     --suite my_suite
   ```
3. **Re‑generate leaderboard summaries** so redactions propagate:

   ```bash
   helm-summarize --suite my_suite --schema schema_medhelm.yaml
   ```


## References

* [Stanford HAI Article](https://hai.stanford.edu/news/holistic-evaluation-of-large-language-models-for-medical-applications)
* [MedHELM Website (latest leaderboard)](https://crfm.stanford.edu/helm/medhelm/latest/)
* [ArXiv Preprint](https://arxiv.org/abs/2505.23802)
