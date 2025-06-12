# MedHELM: Holistic Evaluation of Large Language Models for Medical Applications

**MedHELM** is an extension of the HELM framework for evaluating **large language models (LLMs) in medical applications**.

## Clinician-Validated Taxonomy

MedHELM evaluates models across a clinician-validated taxonomy comprised of **5 categories**, **22 subcategories** and **121 tasks**:

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

This categorization ensures that evaluations reflect the **complexity, diversity, and stakes** of medical AI applications—from assisting clinicians in making critical decisions to safely interacting with patients.

## References

- [Stanford HAI Article](https://hai.stanford.edu/news/holistic-evaluation-of-large-language-models-for-medical-applications)
- [MedHELM Website](https://crfm.stanford.edu/helm/medhelm/latest/)
- [Publication](https://arxiv.org/abs/2505.23802)

## Installation

First, follow the [installation instructions](installation.md) to install the base HELM framework.

To install MedHELM-specific dependencies:

```sh
pip install "crfm-helm[medhelm]"
```

## Getting Started

The following is an example of evaluating [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the [MedCalc-Bench scenario](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/medcalc_bench_scenario.py) using 10 instances.

```sh
helm-run \
  --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct \
  --suite my-medhelm-suite \
  --max-eval-instances 10
```

## Benchmark Access Levels

MedHELM includes a variety of benchmarks that differ in terms of data access requirements. Understanding the type of access required for each benchmark is essential for registering run entries and reproducing results.

### 🔓 Public Benchmarks

These benchmarks are fully open and freely available to the public (e.g., benchmarks hosted on [HuggingFace Datasets](https://huggingface.co/datasets)).

- **Access requirements**: None
- **Example sources**: HuggingFace, GitHub
- **Run entries file**: [run_entries_medhelm_public.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_public.conf)
- **Reproducibility**: Anyone can reproduce results from these benchmarks.

---

### 🔐 Gated Benchmarks

These benchmarks are publicly available but require special permissions or credentials to access (e.g., benchmarks hosted on [PhysioNet](https://physionet.org/)).

- **Access requirements**: User registration, credential verification, or data use agreement
- **Example sources**: PhysioNet, Redivis
- **Run entries file**: [run_entries_medhelm_gated.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_gated.conf)
- **Reproducibility**: Only users with the appropriate access can reproduce results from these benchmarks.

---

### 🔒 Private Benchmarks

These benchmarks are based on proprietary or restricted datasets that are only available to specific organizations.

- **Access requirements**: Organization-specific authorization
- **Example use case**: Internal clinical datasets available only to the originating institution
- **Run entries file**: `run_entries_medhelm_private_{organization}.conf` (_e.g., `run_entries_medhelm_private_stanford.conf`_)
- **Reproducibility**: Only authorized users within the organization can reproduce results from these benchmarks.

---

When contributing or reproducing results, ensure that you are using the correct run entries file corresponding to the benchmark’s access level.

## Reproducing the Leaderboard

To reproduce the [MedHELM leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/), refer to the detailed steps in the [Reproducing Leaderboards](reproducing_leaderboards.md) documentation.

> **Note:** The ability to fully reproduce the leaderboard depends on your access to the underlying benchmarks.
> See [Benchmark Access Levels](#benchmark-access-levels) for details on how access impacts which configuration files to use.

- **Public benchmarks**:  
  Everyone can reproduce the **public portion** of the leaderboard using benchmarks that are freely available (e.g., on HuggingFace). These entries are defined in [run_entries_medhelm_public.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_public.conf).

- **Gated-access benchmarks**:  
  Reproducing results on these benchmarks (e.g., EHRSHOT) requires credentials or approval. These are listed in [run_entries_medhelm_gated.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_gated.conf).

- **Private benchmarks**:  
  These results are specific to certain organizations and require access to private data. They are defined in files following the format `run_entries_medhelm_private_{organization}.conf` under `src/helm/benchmark/presentation/`.

As such, **only users with the necessary access credentials or data permissions** will be able to reproduce the full leaderboard. However, **anyone can reproduce and evaluate the public subset** to benchmark their models.

## Contributing to MedHELM

We welcome contributions from both the research and clinical communities to expand MedHELM with new scenarios and models.

### Scenario Contributions

To contribute a new benchmark scenario to MedHELM, follow the steps below:

#### 1. Create a New Scenario

Start by adding a new scenario class under the [scenarios directory](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios). This class should transform your dataset into HELM’s standardized input format. Your implementation should define:

- The prompt (with any context like patient note, if present)
- The gold standard response(s) (if any)
- Any relevant metadata for post-processing

For detailed guidance, see the [Adding New Scenarios](adding_new_scenarios.md) documentation.

#### 2. Define a Run Spec

Next, define a run specification for your scenario in the [medhelm_run_specs.py](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run_specs/medhelm_run_specs.py) file. This links your scenario to HELM’s command-line interface, and specifies:

- The instruction models should follow
- The metrics to compute
- Any optional annotation steps

##### a. Response Annotations

If your benchmark requires additional processing of model outputs before metric evaluation (e.g., scoring with a LLM-as-a-judge), add an annotation step. This step is executed after obtaining the model responses.

See this [example annotator](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/annotation/med_dialog_annotator.py) for reference.

##### b. Custom Metrics

If your scenario needs a custom evaluation metric not currently supported in HELM, you can define one under the [metrics directory](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/metrics).

See this [example metric implementation](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/metrics/med_dialog_metrics.py) for guidance.

#### 3. Register Your Run Entries

Finally, register your `(model, benchmark)` combinations for the MedHELM leaderboard in the appropriate configuration file based on the type of dataset access.  

- If the benchmark is **fully public** (i.e., doesn't require any credentialized access), add your entry to [run_entries_medhelm_public.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_public.conf).

- If the benchmark requires **gated access** (e.g., PhysioNet datasets or others requiring credentials), use [run_entries_medhelm_gated.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm_gated.conf).

- If the benchmark is **private** to a specific organization, use a dedicated config file under `src/helm/benchmark/presentation/` following the format: `run_entries_medhelm_private_{organization}.conf` (_e.g., `run_entries_medhelm_private_stanford.conf`_)

Each entry specifies a `(model, benchmark)` pair to be evaluated and displayed on the MedHELM leaderboard.

#### 4. Register Your Scenario in Schema MedHELM

To have your scenario appear on the leaderboard, it must be included in the [schema_medhelm.yaml](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/static/schema_medhelm.yaml) file. To do this:

- **Add your scenario name** under the appropriate category in the task taxonomy.
- **Include any new metrics** used by your scenario that are not already listed in the schema.

For reference, see how the `medcalc_bench` scenario is defined in the schema.

### Model Contributions

To contribute with a new model, follow the steps in the [Adding New Models](adding_new_models.md) page.

### Sharing Results

If your contributed benchmark scenarios are **gated** or **private** and you would like to share your results with us, you must redact any sensitive data contained in the prompts and annotations (if applicable). Follow these steps to properly redact and prepare your results:

#### 1. Redact prompt content using the provided script

Run the [redact_scenario_states.py](https://github.com/stanford-crfm/helm/blob/main/scripts/redact_scenario_states.py) script on your `benchmark_output` directory for a particular suite. This script will redact all prompts and responses present under the specified suite. Here’s a usage example:

```sh
python3 scripts/redact_scenario_states.py \
  --output benchmark_output \
  --suite my_suite \
  --redact-output
```

#### 2. Redact annotations (if applicable)

If your scenario includes annotations (e.g., LLM-as-a-judge outputs or any metadata containing sensitive information), ensure these are redacted as well. The annotations are present in the `scenario_state.json` files. You can skip this step if no annotations are included as part of your scenario.

#### **3. Propagate redacted data to leaderboard outputs**

Use helm-summarize to update the leaderboard summaries with the redacted content:

```sh
helm-summarize --suite my_suite --schema schema_medhelm.yaml
```

> **Note:** This step is especially important if you previously ran `helm-summarize` _before_ redacting your outputs.  
> The command copies prompts and responses into the leaderboard files, so rerunning it ensures only redacted content is displayed.
