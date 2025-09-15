# MedHELM: Holistic Evaluation of Large Language Models for Medical Applications

**Who it’s for:** Data scientists at health systems benchmarking LLMs for medical use cases.

**What you’ll do here:** Install MedHELM, run a small evaluation locally, understand access levels, view leaderboards, and learn how to contribute new scenarios/models.

**Time required:** ~10 minutes for the Quickstart.

MedHELM extends the HELM framework to evaluate **large language models (LLMs) in medical applications**, focusing on realistic tasks, safety, and reproducibility.

---

## Requirements 

1. [Conda](https://www.anaconda.com/docs/getting-started/getting-started) (~30 min to intall)
  
    Used to manage the Python virtual environment for MedHELM.

2. [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (~30 min to install)

    Required for downloading MedHELM results stored on Google Cloud.

    **Note:** During installation, the installer will recommend Python 3.12 for full feature support. These features are not needed for MedHELM. We recommend using Python 3.10 for best compatibilitywith MedHELM.

3. GPU Access

    Needed for running models locally. The exact GPU requirements depend on the model and context length.

---

## Quickstart (10 minutes)

**Goal:** Install MedHELM, run a 10-instance evaluation on a public scenario, and open a local leaderboard.

#### 1. Install (Python 3.10 recommended)

Run the following commands to create and activate a new python virtual environment:

```
# Create and activate a clean environment
conda create -n crfm-helm python=3.10 pip
conda activate crfm-helm
pip install -U setuptools
```

Run the following command to install HELM:
```
pip install -U crfm-helm
```

Run the following commands to install the necessary MedHELM extensions:

```
# Install MedHELM extras
pip install "crfm-helm[summarization,medhelm]"
```

#### 2. Run a tiny evaluation

```bash
helm-run \
  --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct \
  --suite my-medhelm-suite \
  --max-eval-instances 10 \
  --output-path ./benchmark_output
```

#### 3. Build and open the local leaderboard

```bash
curl -L -o schema_medhelm.yaml \
  https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/static/schema_medhelm.yaml
helm-summarize --suite my-medhelm-suite --schema schema_medhelm.yaml
helm-server --suite my-medhelm-suite
```

**Expected outcome:**

* A local URL for the leaderboard (printed in the terminal)
* Benchmark results at `./benchmark_output/runs/my-medhelm-suite/`

> ⚠️ **PHI & compliance:** Only run **gated** or **private** data on infrastructure approved by your organization. Use the **redaction** steps in *Sharing Results* before sending outputs externally.

---

## Core Concepts

* **Scenario:** Dataset + prompt/response formatting logic.
* **Run entry:** A `(model, scenario)` pair given to `helm-run`.
* **Suite:** Named collection of runs; appears as a tab/section in the leaderboard.
* **Annotator:** Optional post‑processing (e.g., LLM‑as‑a‑judge).
* **Schema:** Task taxonomy + metrics configuration powering `helm-summarize` and the UI.
* **Release:** The version of leaderboard results.

---

## Clinician‑Validated Taxonomy (overview)

MedHELM evaluates models across a clinician‑validated taxonomy: **5 categories**, **22 subcategories**, **121 tasks**.

- **Clinical Decision Support**
    - Supporting Diagnostic Decisions
    - Planning Treatments
    - Predicting Patient Risks and Outcomes
    - Providing Clinical Knowledge Support

- **Clinical Note Generation**
    - Documenting Patient Visits
    - Recording Procedures
    - Documenting Diagnostic Reports
    - Documenting Care Plans

- **Patient Communication and Education**
    - Providing Patient Education Resources
    - Delivering Personalized Care Instructions
    - Patient-Provider Messaging
    - Enhancing Patient Understanding and Accessibility in Health Communication
    - Facilitating Patient Engagement and Support

- **Medical Research Assistance**
    - Conducting Literature Research
    - Analyzing Clinical Research Data
    - Recording Research Processes
    - Ensuring Clinical Research Quality
    - Managing Research Enrollment

- **Administration and Workflow**
    - Scheduling Resources and Staff
    - Overseeing Financial Activities
    - Organizing Workflow Processes
    - Care Coordination and Planning

---

## Installation

#### 1. Create a virtual environment (Python 3.10 recommended)

Run the following commands to create and activate a new python virtual environment:

```bash
# Create and activate a clean environment
conda create -n crfm-helm python=3.10 pip
conda activate crfm-helm
```

#### 2. Install HELM in your virtual environment:

```bash
pip install crfm-helm
```

#### 3. Install the MedHELM-specific dependencies 

On the same python virtual environment from step 1, run the following command to install the MedHELM-specific dependencies.

```bash
pip install "crfm-helm[summarization,medhelm]"
```

---

## Run Your First Evaluation

The example below evaluates **Qwen2.5‑7B‑Instruct** on the **MedCalc‑Bench** scenario using 10 instances.


#### 1. Run the benchmark 

The following command runs **MedCalc-Bench** on **Qwen2.5‑7B‑Instruct** for 10 instances and stores the results under `./benchmark_output/runs/my-medhelm-suite`. 

```bash
helm-run \
   --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct \
   --suite my-medhelm-suite \
   --max-eval-instances 10 \
   --output-path ./benchmark_output
```

*Flags explained:*

* `--run-entries`: Specifies which benchmark(s) and model(s) to run.
In this example, the benchmark is medcalc_bench and the model is qwen/qwen2.5-7b-instruct, deployed via Hugging Face.
* `--suite`: Assigns the run to a suite, which acts as a collection of results.
Using the same suite name across multiple runs groups their results together.
* `--max-eval-instances`: Limits the number of benchmark instances evaluated. 
* `--output-path`: Path to the directory where output files and results will be written.
If the directory does not exist, it will be created automatically.

#### 2. Create the leaderboard

The following commands convert the results from step 1 into an interactive leaderboard.

```bash
curl -L -o schema_medhelm.yaml \
   https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/static/schema_medhelm.yaml
helm-summarize \
  --suite my-medhelm-suite \
  --schema ./schema_medhelm.yaml \
  --output-path ./benchmark_output
```

*Flags explained:*

* `--suite`: Names the suite where results from this leaderboard creation will be stored.
This allows grouping multiple runs under a common label (e.g., my-medhelm-suite).
* `--schema`: Path to the schema file.
* `--output-path`: Path to the directory where benchmark results from `helm-run` are stored.

#### 3. Run the leaderboard locally

This command runs the leaderboard on a local server. The exact address and port will show on the command output.

```bash
helm-server \
  --suite my-medhelm-suite \
  --output-path ./benchmark_output
```

*Flags explained:*

* `--suite`: Names the suite where results are stored.
This allows grouping multiple runs under a common label (e.g., my-medhelm-suite).
* `--output-path`: Path to the directory where benchmark results from `helm-run` are stored.

---

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

When contributing or reproducing results, ensure you’re using the correct file for the benchmark’s access level.

For **private** benchmarks and organization‑specific configurations, contact the MedHELM team at **[migufuen@stanford.edu](mailto:migufuen@stanford.edu)**.

---

## Viewing and Reproducing Leaderboard Results

You can interact with MedHELM results by **viewing** pre‑computed results locally or by **reproducing** evaluations from scratch.

### View the official leaderboard locally

#### 1. Download raw results
   
Follow the instructions under [Downloading Raw Results](downloading_raw_results.md) up until the *Download a whole project* section to download the MedHELM leaderboard. Make sure to specify the GCS path corresponding to **MedHELM** when downloading the results.

#### 2. Launch the local leaderboard
   
Run the following command to launch the MedHELM leaderboard locally. Use the numbered `release` version you want to display. Check out all release versions on the upper right corner of the official [leaderboard website](https://crfm.stanford.edu/helm/medhelm/latest).

```bash
# Sample command to launch the MedHELM leaderboard version 2.0.0.
helm-server --release v2.0.0
```

### Reproduce leaderboard results

* **Public benchmarks:** Anyone can reproduce the **public subset** using entries in `run_entries_medhelm_public.conf`.
* **Gated benchmarks:** Require credentials/approval (e.g., EHRSHOT); entries live in `run_entries_medhelm_gated.conf`.
* **Private benchmarks:** Org‑specific; entries follow `run_entries_medhelm_private_{organization}.conf`.

> **Note:** The `model_deployments` of the models listed in these run entries are specific to Stanford Healthcare, please change them for the appropriate deployments as needed. For more information on model_deployments, refer to [Adding New Models](adding_new_models.md).

---

## Adding a New Benchmark

MedHELM allows you to define and run custom LLM benchmarks by combining a few simple configuration files:

1. **Prompt template (`.txt`)** — contains the instructions shown to the model, with placeholders (e.g., `{column_name}`) that will be filled in from the dataset.  
2. **Dataset (`.csv`)** — provides the data for each benchmark instance. Each row populates the prompt template and must include required columns like `correct_answer`.  
3. **Benchmark config (`.yaml`)** — specifies the benchmark metadata (name, description), file paths (prompt, dataset), evaluation settings (e.g., max tokens), and the list of metrics to compute.

> **Note:** This feature requires **crfm-helm >= 0.5.8**.  
> Make sure you upgrade before continuing:  
```bash
pip install --upgrade crfm-helm
```

---

### Benchmark Components

#### 1. Prompt Template (`.txt`)

A text file containing the task instructions with placeholders for dataset fields.  

- Placeholders are written as `{column_name}`.  
- At runtime, they will be replaced with values from the dataset.  
- Only columns explicitly referenced in the template will appear in the final prompt.  

Example (`exact_match_demo_prompt.txt`):  
```txt
You are a clinical assistant. Your task is to answer the following medical question based on the patient history.

Patient ID: {patient_id}

Patient Note:
{note}

Question:
{question}

Answer with a single token: "Yes" or "No"
```

#### 2. **Dataset** (`.csv`)

A CSV file with one row per benchmark instance.

- Must contain all columns referenced in the prompt.  

- Must include the following columns:
    - `correct_answer`: The reference answer against which model outputs are compared. 
    - `incorrect_answers`: JSON-serialized list of incorrect alternatives.

Example format:  

```csv
"[""incorrect 1"", ""incorrect 2""]"
```


Example snippet (`exact_match_demo_dataset.csv`):  
```csv
patient_id,note,question,correct_answer,incorrect_answers
001,"Patient with hypertension.","Does the patient have hypertension?","Yes","[""No""]"
002,"Patient denies smoking.","Does the patient smoke?","No","[""Yes""]"
```

#### 3. **Benchmark Config** (`.yaml`)

The benchmark config file must contain the following details of the benchmark:

1. **Name:** A unique identifier for the benchmark.
2. **Description:** A concise description of what the benchmark measures.
3. **Prompt Template Path:** The path to the prompt template file.
4. **Dataset Path:** The path to the csv dataset.
5. **Max tokens:** The maximum number of tokens allowed for model responses. 
6. **Metrics:** A list of all metrics to evaluate and their required arguments (if any). The first metric in the list will be considered the **main metric** for the leaderboard.

Example (`exact_match_demo.yaml`):

```yaml
# Name of your benchmark
name: EXACT-MATCH-DEMO

# Description of your benchmark
description: EXACT-MATCH-DEMO measures LLMs performance on answering yes/no questions based on clinical notes.

# Path to the prompt template
prompt_file: exact_match_demo_prompt.txt

# Path to the dataset
dataset_file: exact_match_demo_dataset.csv

# Max tokens to generate
max_tokens: 1

# Metrics list
# The main metric will be the first one in the list.
metrics:
  - name: exact_match
```


##### Supported Metrics

| Category                | Metric(s)                  | Description                                                                 | Requirements      |
|--------------------------|----------------------------|-----------------------------------------------------------------------------|-------------------|
| **Accuracy Metrics**     | `exact_match`             | Checks if the model output matches the reference exactly.                   | CPU only (light)  |
| **Summarization Metrics**| `rouge_1`, `rouge_2`, `rouge_l` | N-gram overlap metrics commonly used for text comparison.                  | CPU only          |
|                          | `BERTScore-P`, `BERTScore-R`, `BERTScore-F` | Precision, Recall, and F1 based on semantic similarity.                     | GPU required      |
| **LLM-as-Judge Metrics** | `jury_score`              | Uses one or more judge models (LLMs) to evaluate correctness/quality.       | LLM API or GPU    |

---

### Steps to Add Your Own Benchmark

We’ll illustrate each step with `EXACT-MATCH-DEMO`, a benchmark that measures LLMs performance on answering yes/no questions based on clinical notes.

#### 1. Create a prompt template (`.txt`)  

Write the task instructions in plain text, and use `{column_name}` placeholders wherever you want values from the dataset to be inserted. Every placeholder must match a column name in your dataset exactly.

**Example — `exact_match_demo_prompt.txt`:**
```txt
You are a clinical assistant. Your task is to answer the following medical question based on the patient history.

Patient ID: {patient_id}

Patient Note:
{note}

Question:
{question}

Answer with a single token: "Yes" or "No"
```

#### 2. Prepare a dataset (`.csv`)  

Build a CSV file where each row represents one benchmark instance.  

- Include all columns referenced in your prompt template.  
- Add a `correct_answer` column (required).  
- Add an `incorrect_answers` column containing a JSON-serialized list of alternative answers.

**Example — `exact_match_demo_dataset.csv`:**

```csv
patient_id,note,question,correct_answer,incorrect_answers
001,"Patient with hypertension.","Does the patient have hypertension?","Yes","[""No""]"
002,"Patient denies smoking.","Does the patient smoke?","No","[""Yes""]"
```

#### 3. Write a benchmark config (`.yaml`)  

Create a YAML file that defines:

- `name:` The name of the benchmark. This will appear on the leaderboard.
- `description:` A short description of the benchmark. This will also be shown on the leaderboard.
- `prompt_file:` Path to the prompt template file for your benchmark.
- `dataset_file:` Path to the dataset file for your benchmark.
- `max_tokens:` The maximum number of tokens that models are allowed to generate in response to your benchmark prompts. Choose this value based on the expected response length.
- `metrics:` A list of evaluation metrics to be used for your benchmark.

**Example — `exact_match_demo.yaml`:**

```yaml
# Name of your benchmark
name: EXACT-MATCH-DEMO

# Description of your benchmark
description: EXACT-MATCH-DEMO measures LLMs performance on answering yes/no questions based on clinical notes.

# Path to the prompt template
prompt_file: exact_match_demo_prompt.txt

# Path to the dataset
dataset_file: exact_match_demo_dataset.csv

# Max tokens to generate
max_tokens: 1

# Metrics list
# The main metric will be the first one in the list.
metrics:
  - name: exact_match
```

#### 4. Define a run configuration (`.conf`)  

List the models you want to evaluate on your benchmark. Each entry specifies a model, its deployment, and the path to your YAML config.

**Example — `exact_match_demo.conf`:**

```yaml
entries: [
  {description: "medhelm_configurable_benchmark:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct,config_path=exact_match_demo.yaml", priority: 1},
]
```

**Note:**  All benchmarks defined this way must begin with the prefix `medhelm_configurable_benchmark`.  
The only part that changes between benchmarks is the `config_path`, which should point to the YAML config file for the specific benchmark you want to run.

#### 5. Run the benchmark  

With your prompt template, dataset, YAML config, and run configuration ready, you can now execute the benchmark.  

First, set a few environment variables to control the run:  

```bash
# Name of the benchmark suite (used to group results)
export SUITE_NAME=DEMO

# Path to your run configuration file (.conf)
export RUN_ENTRIES_CONF_PATH="./exact_match_demo.conf"

# Maximum number of evaluation instances to run (useful for quick testing)
export MAX_EVAL_INSTANCES=2
```

Then, run the benchmark with:

```bash
helm-run \
  --conf-paths $RUN_ENTRIES_CONF_PATH \
  --max-eval-instances $MAX_EVAL_INSTANCES \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

#### 6. Recreate the leaderboard

With the benchmarks results from step 5, run the following command to recreate the leaderboard:

```bash
helm-summarize \
  --auto-generate-schema \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

#### 7. Launch the leaderboard

Once the leaderboard is created, launch the leaderboard to see it live.

```bash
helm-server \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

---

### Advanced Example: Jury Score Benchmark

In some healthcare tasks, standard accuracy or overlap metrics (like `exact_match` or `rouge`) may fail to capture the quality of an LLM’s response. To handle these cases, MedHELM supports **rubric-based evaluation**, where a set of LLMs — called *judges* — evaluate model responses according to predefined criteria.  

This example, `LLM-JURY-SCORE-DEMO`, demonstrates how to create a benchmark that uses the `jury_score` metric to evaluate patient-friendly discharge summaries based on clinical notes. The setup is similar to the [Exact Match Demo](#example-1-exact-match-demo-beginner); here we only highlight the differences where necessary.  

---

#### 1. Create a prompt template (`.txt`)

**Example — `llm_jury_score_demo_prompt.txt`:**
```txt
You are a clinical assistant. Your task is to write a plain-language discharge summary for patients.

Patient Note:
{note}

Constraints: {constraints}

Please provide your response below:
```

#### 2. Prepare a dataset (`.csv`)

**Example — `llm_jury_score_demo_dataset.csv`:**
```csv
patient_id,note,constraints
4001,"67-year-old with cough and fever; chest X-ray shows pneumonia. Current meds: metformin 500 mg BID, atorvastatin 20 mg QHS. New Rx: amoxicillin-clavulanate 875/125 mg PO BID x7 days. Return if worse. Follow-up PCP in 3 days.","<=120 words; bullet points allowed; use lay terms; include diagnosis, meds, warning signs, and follow-up."
4002,"58-year-old with CKD stage 3. Meds: metformin 1000 mg BID; ibuprofen stopped; lisinopril 10 mg daily started. Cr 1.8 mg/dL. Counsel to avoid NSAIDs.","<=100 words; use headings; avoid jargon; include medication changes and safety advice."
```

#### 3. Define the benchmark config (`.yaml`)

The key difference in this example is the **metrics list**, where we define a `jury_score` metric. This requires a *judge prompt* and a list of judge models.  

##### 3.1 Judge Prompt (`.txt`)

The judge prompt instructs the judge models on how to evaluate outputs. It must include:  

1. `{QUESTION}` — the original prompt shown to the evaluated model.  
2. `{RESPONSE}` — the evaluated model’s output.  
3. `{GOLD_RESPONSE}` *(optional)* — the gold answer from the dataset.  
4. `<rubric_criteria>` — the evaluation rubric in JSON format, where each criterion is a JSON object with a `score` (0–5) and an explanation.  

**Example — `llm_jury_score_demo_judge_prompt.txt`:**

```txt
You are an expert in clinical communication. Your task is to evaluate the quality of the following discharge summary written for a patient.

<QUESTION>
{QUESTION}
</QUESTION>

The response given to the above question is:

<RESPONSE>
{RESPONSE}
</RESPONSE>

Evaluate the model under the following criteria (assign a score from 0 to 5 for each):

Faithfulness (0 = completely inaccurate; 5 = entirely accurate).  
Completeness (0 = missing required elements; 5 = fully covers diagnosis, meds, follow-up, warning signs).  
Safety (0 = harmful advice; 5 = completely safe).  
Clarity (0 = incomprehensible; 5 = crystal clear for patients).  
Conciseness (0 = too short/verbose; 5 = meets length and avoids repetition).  

Output Format:  
Generate a valid JSON object inside <rubric_criteria>:  

<rubric_criteria>
{
  "faithfulness": { 
    "score": 0,
    "explanation": "Explain why this score was given." 
  },
  "completeness": {
    "score": 0,
    "explanation": "Explain why this score was given."
  },
  "safety": {
    "score": 0,
    "explanation": "Explain why this score was given."
  },
  "clarity": {
    "score": 0,
    "explanation": "Explain why this score was given."
  },
  "conciseness": {
    "score": 0,
    "explanation": "Explain why this score was given."
  }
}
</rubric_criteria>

Ensure the output is valid JSON:
- Use double quotes (") for all keys and values.  
- Escape quotes inside explanations (e.g., \"like this\").  
- Do not include any text outside the JSON.
```

**Notes:** 

1. Judge responses must be a valid JSON in the specified format. Invalid responses are ignored. 
2. Score scales must start at 0.
3. Currently, we only support criteria where a higher value means better performance.

##### 3.2 Selecting Judges

Each judge must be defined with:  
- `name`: The deployment identifier (must match `name` in `model_deployments.yaml`).  
- `model_name`: The model name (must match `model_name` in `model_deployments.yaml`).  

##### 3.3 Full Benchmark Config

With the judge prompt defined (3.1) and the judges selected (3.2), you can now put everything together into the full benchmark configuration file.

**Example — `llm_jury_score_demo.yaml`:**
```yaml
# Name of your benchmark
name: LLM-JURY-SCORE-DEMO

# Description of your benchmark
description: LLM-JURY-SCORE-DEMO measures LLM performance on creating patient-friendly discharge summaries.

# Path to the prompt template
prompt_file: llm_jury_score_demo_prompt.txt

# Path to the dataset
dataset_file: llm_jury_score_demo_dataset.csv

# Max tokens to generate
max_tokens: 1024

# Metrics list
# The main metric will be the first one in the list.
metrics:
  - name: "jury_score"
    # Path to the judge prompt
    prompt_file: llm_jury_score_demo_judge_prompt.txt
    # List of LLM judges
    judges:
      - name: "huggingface/qwen2.5-7b-instruct"
        model_name: "qwen/qwen2.5-7b-instruct"
```

**How the Jury Score is Calculated**  

For each instance, every judge model evaluates the output using the rubric criteria.  

- If your rubric has 4 criteria and you specify 3 judges, there will be **12 individual scores** (4 × 3).  
- The final `jury_score` is the **simple mean** of all these scores.  

This ensures that all judges and all criteria contribute equally to the final evaluation.


#### 4. Define a run configuration (`.conf`)

**Example — `llm_jury_score_demo.conf`:**
```yaml
entries: [
  {description: "medhelm_configurable_benchmark:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct,config_path=llm_jury_score_demo.yaml", priority: 1},
]
```

#### 5. Run the benchmark

```bash
export SUITE_NAME=DEMO
export RUN_ENTRIES_CONF_PATH="./llm_jury_score_demo.conf"
export MAX_EVAL_INSTANCES=2

helm-run \
  --conf-paths $RUN_ENTRIES_CONF_PATH \
  --max-eval-instances $MAX_EVAL_INSTANCES \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

#### 6. Recreate the leaderboard

```bash
helm-summarize \
  --auto-generate-schema \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

---

#### 7. Launch the leaderboard

```bash
helm-server \
  --suite $SUITE_NAME \
  --output-path ./benchmark_output
```

#### Key Differences from the Exact Match Demo
- The **prompt** produces longer, free-text outputs (summaries).  
- The **metric** is `jury_score`, which requires a judge prompt and judge models.  
- The **evaluation** uses LLMs as judges rather than string overlap metrics.

## References

* [Stanford HAI Article](https://hai.stanford.edu/news/holistic-evaluation-of-large-language-models-for-medical-applications)
* [MedHELM Website (latest leaderboard)](https://crfm.stanford.edu/helm/medhelm/latest/)
* [ArXiv Preprint](https://arxiv.org/abs/2505.23802)
