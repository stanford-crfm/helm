# MedHELM (LLM Evaluation in Medical Applications)

**MedHELM** is an extension of the HELM framework for evaluating **large language models (LLMs) in medical applications**.

## Holistic Evaluation of Language Models for Medical Applications

As large language models (LLMs) become increasingly integrated into clinical workflows, it is critical to understand their strengths and limitations across a broad range of **real-world medical tasks**. Existing evaluation benchmarks often focus narrowly on question-answering accuracy or single datasets, which fail to capture the complexity and risks of deploying LLMs in healthcare.

To address this gap, we introduce **MedHELM**, a benchmark for the **Holistic Evaluation of Language Models for Medical Applications**. MedHELM evaluates models across **clinically grounded categories and subcategories** that reflect real-world tasks with real patient data.

### HealtchCare Categories and Subcategories

MedHELM evaluates models across **5 categories** and **22 subcategories**:

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
- [HELM Website](https://crfm.stanford.edu/helm/latest/)
- Publication (Coming Soon)

## Installation

First, follow the [installation instructions](installation.md) to install the base HELM framework.

To install MedHELM-specific dependencies:

```sh
pip install "crfm-helm[medhelm]"
```

## Getting Started

The following is an example of evaluating [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the [MedCalc-Bench scenario](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/medcalc_bench_scenario.py) using 10 instances.

```sh
helm-run --run-entries medcalc_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct --suite my-medhelm-suite --max-eval-instances 10
```

## Reproducing the Leaderboard

To reproduce the [entire MedHELM leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/), refer to the instructions for MedHELM on the [Reproducing Leaderboards](reproducing_leaderboards.md) documentation.

## Contributing to MedHELM

We welcome contributions from both the research and clinical communities to expand MedHELM with new scenarios and models.

### Scenario Contributions

To contribute a new benchmark scenario to MedHELM, follow the steps below:

#### 1. Create a New Scenario

Start by adding a new scenario class under the [scenarios directory](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios). This class should transform your dataset into HELM’s standardized input format. Your implementation should define:

- The prompt context
- The reference response(s)
- Any relevant metadata for post-processing

For detailed guidance, see the [Adding New Scenarios](adding_new_scenarios.md) documentation.

#### 2. Define a Run Spec

Next, define a run specification for your scenario in the [`medhelm_run_specs.py`](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run_specs/medhelm_run_specs.py) file. This links your scenario to HELM’s command-line interface, and specifies:

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

Finally, register your model-scenario combinations in the [`run_entries_medhelm.conf`](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_medhelm.conf) file. Each entry defines a (model, benchmark) pair in the MedHELM leaderboard.

### Model Contributions

To contribute with a new model, follow the steps in the [Adding New Models](adding_new_models.md) page.
