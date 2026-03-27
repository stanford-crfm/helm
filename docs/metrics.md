---
title: Metrics
---
# Metrics

HELM supports a variety of metrics for evaluating model outputs (e.g. accuracy, efficiency, bias, toxicity). The metrics are implemented in the `helm.benchmark.metrics` module in the source code.

For the full API reference, build the documentation with MkDocs from the repository root (`pip install -e . && mkdocs serve`) or see the [source code]({{ site.repo_url }}/tree/main/src/helm/benchmark/metrics).


## air_bench_metrics

### AIRBench2024BasicGenerationMetric()

Replacement for BasicGenerationMetric for AIRBench 2024.

We call compute_request_state_metrics here because we can't use `BasicGenerationMetric` because we abuse "references" to store metadata rather than true metadata.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

### AIRBench2024ScoreMetric

Score metrics for AIRBench 2024.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## annotation_metrics

### AnnotationLabelMetric(annotator_name: str, key: str, labels: List[str])

Binary metric for labels produced by annotators.

Expects the annotation with the given annotator name and key to be a string label.

For each possible label in the list of possible labels, produces a corresponding stat with a value of 1 or 0 indicating if the actual label in the annoation.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

### AnnotationLikertScaleMetric(annotator_name: str, key: str, min_score: int, max_score: int)

Numeric metric for labels produced by annotators.

Expects the annotation with the given annotator name and key to be a string label.

For each possible label in the list of possible labels, produces a corresponding stat with a value of 1 or 0 indicating if the actual label in the annoation.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

### AnnotationNumericMetric(annotator_name: str, key: str)

Numeric metric for numbers produced by annotators.

Expects the annotation with the given annotator name and key to be a number.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## basic_metrics

### BasicGenerationMetric(names: List[str])

Defines basic metrics which don't require domain knowledge. This should be fairly comprehensive already, and we should try to use this as much as possible. If we need a different variant, try to generalize this or factor things out. It's possible we don't need to subclass this. `names` is a list of optional metrics to be specified by the user. Currently only `exact_match` is supported.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Compute all metrics.

### BasicReferenceMetric()

Defines basic metrics for Scenarios that use one Request per Reference instead of one per Instance.

#### evaluate_references(adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Perform evaluation when we have made different requests for each reference. For each reference, we have a model score (log probability) and whether it's correct.

### InstancesPerSplitMetric

Report the average num_instances in each MetricContext across train_trials.

### _compute_finish_reason_metrics(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Record how often generation finished due to reaching token limit, stop token(s), or end of text

### _compute_truncation_metrics(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Record the number of training instances used in the prompt and whether even the prompt needed to be truncated (once we hit zero training instances).

### compute_calibration_metrics(per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]

### compute_language_modeling_metrics(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Compute the logprob and normalization factors for the first completion

### compute_perplexity_metrics(stats: Dict[MetricName, Stat]) -> List[Stat]

### compute_request_state_metrics(efficiency_metric: EfficiencyMetric, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Compute metrics that are common to both `evaluate_generation` and `evaluate_references`.

## bbq_metrics

### BBQMetric

Defines metrics for the BBQ dataset. Specifically, we are interested in the following metrics:

- the bias score in disambiguated contexts (self.disamb_bias_score), where it is possible to give a definitely correct answer
- the bias score in ambiguous contexts (self.amb_bias_score), where it's NOT possible to give a definitively correct answer (and so the correct answer is "Unknown")

Other terms:

- target: when the model predicts the stereotyped group in question as the answer
- non_target: when the model predicts a non-stereotyped group as the answer
- neg: when the question is negative (e.g., "who is bad at math?")
- non_neg: when the question isn't negative (e.g., "who is good at math?")
- unknown: when the model answers "unknown" / some variation of "unknown"
- non_unknown: when the model answers anything else
- acc: accuracy, or the percentage of questions answered correctly

The bias scores reflect the percentage of non-unknown responses that align with social biases; i.e., answering with a member of a stereotyped group for a negative question and answering with a non-member of a stereotyped group for a non-negative question.

Specifically:

- self.amb_bias_score = ((2 * (self.amb_target_and_neg + self.amb_non_target_and_non_neg) / self.amb_non_unknown) - 1) * (1 - acc)
- self.disamb_bias_score = (2 * (self.disamb_target_and_neg + self.disamb_non_target_and_non_neg) / self.disamb_non_unknown) - 1

For more details, see the equation on page 6 of [https://arxiv.org/pdf/2110.08193.pdf](https://arxiv.org/pdf/2110.08193.pdf)

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## bias_metrics

### BiasMetric(mode: str, demographic_category: str, target_category: Optional[str] = None)

Compute metrics to evaluate social bias.

We compute demographic representation and mean stereotypical association bias in model generated text using word counts and co-occurrences. Refer to the documentation for the following methods for more information:

- `evaluate_demographic_representation`
- `evaluate_stereotypical_associations`

References:

1. Garg et al. 2018 | [https://arxiv.org/abs/1711.08412](https://arxiv.org/abs/1711.08412)
2. Bolukbasi et al. 2016 | [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520)

mode (`str`) –

Method used to compute the bias score, one of "representation" or "associations". The latter also requires `target_category` to be provided. Following methods are called depending on the mode: - self.evaluate_demographic_representation: The method used to compute the bias score if the "representation" mode is selected. - self.evaluate_stereotypical_associations: The method used to compute the bias score if the "associations" mode is selected.

demographic_category (`str`) –

The demographic category for which the bias score will be computed, one of "race" or "gender".

target_category (`Optional[str]`, default:`None`) –

The target category used to measure the stereotypical associations with the "demographic_category". One of "adjective" or "profession".

| Parameters: |
| --- |

#### evaluate_demographic_representation(texts: List[str]) -> Optional[float]

Compute the score measuring the bias in demographic representation.

The steps to compute the bias score are as follows:

1. Create a count vector for all the demographic groups by:
   - Getting the list of words for each demographic group;
   - Counting the number of total times words in a specific group's list occur in "texts".
2. Compute the bias score following the steps in self.group_counts_to_bias.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

Compute the bias score on the request_states.

#### evaluate_stereotypical_associations(texts: List[str]) -> Optional[float]

Compute the mean stereotypical association bias of the target words and demographic groups.

Once we get the list of target words and groups for the specified target_category and demographic_group, respectively, we compute the mean bias score as follows:

1. For each text in texts, count the number of times each target word in the target word list co-occur with a word in the demographic's word list.
2. Compute a bias score for each target word following the steps in self.group_counts_to_bias.
3. Take the mean of the bias scores, which corresponds to the extent the average association of different groups with the target terms in model-generated text diverges from equal representation.

## bigcodebench_metrics

### BigCodeBenchMetric

Score metrics for BigCodeBench.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## bird_sql_metrics

### BirdSQLMetric

Score metrics for Bird-SQL.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## classification_metrics

### ClassificationMetric(averages: Optional[List[Optional[str]]] = None, labels: Optional[List[str]] = None, scores: Optional[List[str]] = None, delimiter: Optional[str] = None)

Defines metrics for multi-class classification using the generation adapter.

Currently provides `classification_macro_f1` and `classification_micro_f1`. These are population-level F1 measures to measure classification performance where each generation is a predicted class, and are different from the instance-level F1 measures in `BasicMetrics` that are intended to measure word overlap between the correct references and generations. The correct class should be provided by the normalized text of a correct reference. The predicted class for each instance is the normalized text of the generation.

Note: - It is highly recommended to specify the set of classes should be specified using the `labels` parameter. Otherwise, the set of classes is derived from the correct references from all the instances. This means that classes may be incorrectly omitted if they are never used as a correct reference. - The `averages` parameter is a list of averaging methods to be used. It has the same meaning `average` as in scikit-learn. - Generations that are not in any of the known classes are counted as a negative prediction for every class. - Perturbed classes are considered different classes from unperturbed classes. - Currently, multi-label classification is not supported.

:param delimiter: For multi-label classification, the string delimiter between classes in the model's output. :param average: The list of scores to compute (e.g. "f1", "precision", "recall"). Defaults to ["f1"]. :param average: The averaging methods (e.g. "micro", "macro", "weighted") to be used. It has the same meaning `average` as in scikit-learn. Defaults to ["macro", "micro"]. :param labels: The set of labels. :return: A list of `Stat` objects.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

### MultipleChoiceClassificationMetric

Calculate population micro/macro F1 score for multiple_choice_* adapters. For generation adapters, please use ClassificationMetric.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## cleva_accuracy_metrics

### CLEVATopKAccuracyMetric(k: int, cut_off: int)

Defines metrics for CLEVA conceptual generalization task.

This is not a conventional accuracy@k metric but rather a special one taken from [https://openreview.net/pdf?id=gJcEM8sxHK](https://openreview.net/pdf?id=gJcEM8sxHK)

It accepts multiple predictions and multiple references to calculate the accuracy per instance. For each instance, the model gets perfect accuracy as long as the substring of any reference appears in the first few tokens in one of the prediction.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## cleva_harms_metrics

### CLEVABiasMetric(mode: str, demographic_category: str, target_category: Optional[str] = None)

Compute metrics to evaluate social bias in Chinese.

The implementation is inherited from [https://github.com/PacificAI/medhelm/blob/main/src/helm/benchmark/metrics/bias_metrics.py](https://github.com/PacificAI/medhelm/blob/main/src/helm/benchmark/metrics/bias_metrics.py)

mode (`str`) –

Method used to compute the bias score, one of "representation" or "associations". The latter also requires `target_category` to be provided. Following methods are called depending on the mode: - self.evaluate_demographic_representation: The method used to compute the bias score if the "representation" mode is selected. - self.evaluate_stereotypical_associations: The method used to compute the bias score if the "associations" mode is selected.

demographic_category (`str`) –

The demographic category for which the bias score will be computed, one of "race" or "gender".

target_category (`Optional[str]`, default:`None`) –

The target category used to measure the stereotypical associations with the "demographic_category". One of "adjective" or "profession".

| Parameters: |
| --- |

#### evaluate_demographic_representation(texts: List[str]) -> Optional[float]

Code is mainly inherited from the parent class except for modification of word segmentation.

#### evaluate_stereotypical_associations(texts: List[str]) -> Optional[float]

Code is mainly inherited from the parent class except for modification of word segmentation.

### CLEVACopyrightMetric(name: str, normalize_by_prefix_length: str = False, normalize_newline_space_tab: str = False)

Basic copyright metric for Chinese.

### CLEVAToxicityMetric

Toxicity metric for Chinese.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Code is mainly inherited and only Chinese language is added to API requests.

## code_metrics

Evaluating source code generation.

### APPSMetric(names, timeout)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## codeinsights_code_efficiency_metrics

### CodeInsightsCodeEfficiencyMetric(num_runtime_runs: int = 5, timeout_seconds: int = 10)

Comprehensive metric combining functional correctness and runtime efficiency evaluation.

This metric first evaluates functional correctness and then measures runtime efficiency alignment between LLM-generated code and student reference code when both are correct.

timeout (`int`) –

Timeout for each test case execution.

| Parameters: |
| --- |

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate LLM-generated code by running unit tests and computing pass rate.

**Returns:** List[[Stat](https://crfm-helm.readthedocs.io/en/latest/schemas/#helm.benchmark.metrics.statistic.Stat)] — List of Stat objects containing the functional correctness score.


## codeinsights_code_evaluation_metrics

### AdvancedCodeEvaluationMetric(use_codebert: bool = True)

Extended code evaluation metric with additional analyses

### CodeInsightsCodeEvaluationMetric(use_codebert: bool = True)

Metric for evaluating code generation quality using AST analysis and CodeBERT similarity.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate a single generated code snippet.

### CodeInsightsComprehensiveCodeEvaluationMetric(use_codebert: bool = True)

Comprehensive metric combining AST, CodeBERT, and unit test alignment.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate with AST, CodeBERT, and unit test alignment metrics.

### UnitTestAlignmentMetric

Metric for evaluating C++ code generation by comparing unit test results with student correctness pattern.

#### _calculate_alignment_metrics(llm_pattern: List[int], student_pattern: List[int]) -> List[Stat]

Calculate alignment metrics between LLM and student correctness patterns.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate LLM-generated code by running unit tests and computing pass rate.

**Returns:** List[[Stat](https://crfm-helm.readthedocs.io/en/latest/schemas/#helm.benchmark.metrics.statistic.Stat)] — List of Stat objects containing the functional correctness score.


### evaluate_ast_distances_batch(results: Dict, analyzer: ASTAnalyzer) -> pd.DataFrame

Legacy batch evaluation method for AST distances. This can be used outside of HELM if needed.

## codeinsights_correct_code_metrics

### CodeInsightsFunctionalCorrectnessMetric(timeout: int = 10, max_workers: int = 8)

Metric for evaluating functional correctness of C++ code generation.

Measures each model's functional correctness by computing the proportion of problems for which its generated code passes all provided unit tests. For every generated solution, we compile the C++ code (using g++) and execute the full test cases. We record the proportions of the unit test that passes for each problem and then take the mean across all problems. This yields a score between 0 and 1, where 1 indicates the model produced flawless codes, and lower values reveal the fraction of tasks it could not solve all the unit test cases.

timeout (`int`, default:`10`) –

Timeout for each test case execution.

max_workers (`int`, default:`8`) –

Maximum number of workers for parallel processing.

| Parameters: |
| --- |

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate LLM-generated code by running unit tests and computing pass rate.

**Returns:** List[[Stat](https://crfm-helm.readthedocs.io/en/latest/schemas/#helm.benchmark.metrics.statistic.Stat)] — List of Stat objects containing the functional correctness score.


## codeinsights_edge_case_metrics

### CodeInsightsUnittestAlignmentMetric(use_codebert: bool = True)

unit-test alignment (with new metrics).

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str)

### UnittestAlignmentMetric

Compare LLM unit-test results with the student’s correctness pattern.

Adds

• functional_correctness (pass-rate) • edge_case_slip_match (binary 0/1)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## conv_fin_qa_calc_metrics

### ConvFinQACalcMetric

Score metrics for AIRBench 2024.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## copyright_metrics

### BasicCopyrightMetric(name: str, normalize_by_prefix_length: str = False, normalize_newline_space_tab: str = False)

Basic copyright metric for evaluating surface-level similarity.

This class supports `longest_common_prefix_length` and `edit_distance`. In contrast to model-based semantic similarity evaluation.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Compute the length of the longest common prefix between reference and generations.

Result is based on number of tokens produced with `nltk.tokenize.TreebankWordTokenizer`. When there are multiple generations, return the length of the longest.

Example:

```
input: A
generations: [A A B C, A M D]
reference: A A D
returns: 2
explanation: The longest common prefix is A A (between A A B C and A A D).

```

## czech_bank_qa_metrics

## decodingtrust_fairness_metrics

### FairnessMetric

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## decodingtrust_ood_knowledge_metrics

### OODKnowledgeMetric

OOD knowledge metric for evaluating the OOD answering reliability. A higher rr and macc indicates a better reliability.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## decodingtrust_privacy_metrics

### PrivacyMetric

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## decodingtrust_stereotype_bias_metrics

### StereotypeMetric

Stereotype bias metric used for the DecodingTrust Stereotype Bias study. Asks LLM to respond to stereotypes with an agreement/disagreement based on its stance. The dataset tests for 3 variations of 16 stereotype topics for 24 demographic groups. Final bias metric is the average agreement of the model with stereotype statements.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## disinformation_metrics

Diversity metrics for the disinformation scenario.

### DisinformationMetric(name)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## dry_run_metrics

### DryRunMetric()

Metrics for dry run.

## efficiency_metrics

### EfficiencyMetric()

#### compute_efficiency_metrics(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Compute efficiency metrics for both inference and training. For inference, we record both the actual runtime and an estimated idealized runtime for the given request with an optimized software implementation run on A100 GPU(s), taking into account both the number of tokens in the prompt of the request, and the number of generated output tokens. For training, we report the estimated total metric tons of CO2 emitted to train the model. This is the same for each request.

## ehr_sql_metrics

### EhrSqlMetric

Metric for evaluating the EHR SQL dataset, focusing on: 1. Execution Accuracy – Whether the generated SQL query produces the same results as the ground truth. 2. Query Validity – Whether the generated SQL executes without errors. 3. Precision for Answerable Questions (Pans). 4. Recall for Answerable Questions (Rans).

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate execution accuracy, query validity, and answerability metrics.

## evaluate_instances_metric

### EvaluateInstancesMetric

Metric that needs to examine all request states for all instances in the same split with the same perturbations in order to determine the Stats.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat] `abstractmethod`

Evaluate all request states directly. Use only if nothing else works.

## evaluate_reference_metrics

### compute_reference_metrics(names: List[str], adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService) -> List[Stat]

Setup:

- Gold (correct references): G1 ... Gm
- Predictions (completions): P1 ... Pk

For each pair (G, P), we can define a ${score} (e.g., exact match, F1, BLEU).

We define the following stats:

- ${score}: max_i score(Gi, P1)
- ${score}@k: max_{i,j} score(Gi, Pj)

## fin_qa_metrics

### FinQAMetric

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## gpt4_audio_critique_metrics

### GPT4AudioCritiqueMetric(num_respondents: int, max_tokens: int)

Critique evaluation for evaluating how original the generated text are given the image by GPT4.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## gpt4_audio_refusal_metrics

### GPT4AudioRefusalCritiqueMetric(num_respondents: int, max_tokens: int)

This metric evaluates the model's refusal to answer jailbreak attack prompts.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## gpt4v_originality_critique_metrics

### GPT4VCritiqueMetric(num_respondents: int)

Critique evaluation for evaluating how original the generated text are given the image by GPT4V.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## ifeval_metrics

### IFEvalMetric

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## instruction_following_critique_metrics

### InstructionFollowingCritiqueMetric(num_respondents: int)

Critique evaluation for instruction following. Possesses the ability to ask human annotators the following questions about the model responses:

1. Response relevance/helpfulness
2. How easy it is to understand the response
3. How complete the response is
4. How concise the response is
5. Whether the response uses toxic language or helps the user with harmful goals
6. Whether all facts cited in the response are true

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Get critiques of a summary and compute metrics based on the critiques.

## kpi_edgar_metrics

### KPIEdgarMetric

Word-level entity type classification F1 score, macro-averaged across entity types.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## language_modeling_metrics

### LanguageModelingMetric(names: List[str])

Defines the basic metrics available when using the ADAPT_LANGUAGE_MODELING adapter. This is parallel to BasicMetric and produces many of the same Stats.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Compute all metrics.

## live_qa_metrics

### LiveQAScoreMetric

Score metrics for LiveQA.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## llm_jury_metrics

### LLMJuryMetric(metric_name: str, scenario_name: str, annotator_models: Dict[str, AnnotatorModelInfo], default_score: float = 0.0)

Score metrics for LLM Jury.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## lmkt_metrics

### SemanticSimilarityMetric(similarity_fn_name: str = 'cosine')

Score metrics for LMKT semantic similarity measurement.

Available options are "dot", "cosine", "manhattan" and "euclidean".

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## machine_translation_metrics

### CLEVAMachineTranslationMetric

Compute the BLEU score for Machine Translation scenarios of CLEVA benchmark. Based on sacrebleu, this implementation distinguishes target language and allows variable number of references. If there are more than one hypothesis, only the first one is adopted in the calculation.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

Compute the corpus-level metric based on all reqeust_states.

### MachineTranslationMetric

Compute the BLEU score for Machine Translation scenarios. The implementation is based on sacrebleu.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

Compute the corpus-level metric based on all reqeust_states.

## medcalc_bench_metrics

### MedCalcBenchMetric

Metric for evaluating the MedCalc Bench dataset, assessing the model's ability to be a clinical calculator.

Exact match based on category: 1. Normal exact match: for categories "risk", "severity" or "diagnosis". 2. Variant exact match: for other categories, if the number calculated by the model falls between the values in the Lower limit and Upper limit columns, we mark it as accurate.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate a single generation against reference labels.

## medec_metrics

### MedecMetric

Metric for evaluating the MEDEC dataset, assessing medical error detection and correction.

- Error Flag Accuracy: Whether the model correctly identifies if a medical note contains an error.
- Error Sentence Detection Accuracy: Whether the model correctly identifies the erroneous sentence when an error is present.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate a single LLM generation against the ground truth labels.

## mimiciv_billing_code_metrics

### MIMICIVBillingCodeMetric

Metric for evaluating the MIMIC Billing Code dataset, assessing the model's ability to match the reference ICD codes. Handles cases where raw prediction output contains additional text.

Calculates: 1. Precision: proportion of correctly predicted ICD codes among all predicted codes 2. Recall: proportion of correctly predicted ICD codes among all reference codes 3. F1 score: harmonic mean of precision and recall

ICD codes format: letter followed by 1-3 digits, optional period, optional additional digits

"J18.9", "J45.909", "J47.1", "J96.01"

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Evaluate a single generation against reference labels.

## omni_math_metrics

### OmniMATHMetric

Score metrics for Omni-MATH.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## openai_mrcr_metrics

### OpenAIMRCRMetric

Accuracy metric for OpenAI MRCR.

The measured metric is the SequenceMatcher ratio as implemented in [https://docs.python.org/3/library/difflib.html](https://docs.python.org/3/library/difflib.html). The model must prepend an alphanumeric hash to the beginning of its answer. If this hash is not included, the match ratio is set to 0. If it is correctly included, the stripped sampled answer is compared to the stripped ground truth answer.

Adapted from: [https://huggingface.co/datasets/openai/mrcr/blob/204b0d4e8d9ca5c0a90bf942fdb2a5969094adc0/README.md](https://huggingface.co/datasets/openai/mrcr/blob/204b0d4e8d9ca5c0a90bf942fdb2a5969094adc0/README.md)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## paraphrase_generation_metrics

### CLEVAParaphraseGenerationMetric(alpha: float = 0.8)

Compute the Chinese iBLEU score for Paraphrase Generation scenarios of CLEVA benchmark. This implementation allows variable number of references (i.e., golds). If there are more than one hypothesis (i.e., preds), only the first one is adopted in the calculation.

Reference: [https://aclanthology.org/2022.acl-long.178.pdf](https://aclanthology.org/2022.acl-long.178.pdf) [https://aclanthology.org/P12-2008.pdf](https://aclanthology.org/P12-2008.pdf)

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## prometheus_vision_critique_metrics

### PrometheusVisionCritiqueMetric(num_respondents: int, max_tokens: int)

We compute the same metrics from the Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation paper: [https://arxiv.org/pdf/2401.06591.pdf](https://arxiv.org/pdf/2401.06591.pdf)

In this paper, the output of a Vision-Language Model named Prometheus-Vision is used to evaluate the quality of the output of other Vision-Language Models to be evaluated.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## ranking_metrics

### RankingMetric(method: str, measure_names: List[str], correct_output: str, wrong_output: str, rank: Optional[int] = None, multiple_relevance_values: bool = False)

Ranking metric.

method (`str`) –

The adaptation method used. The method must exists in self.METHOD_LIST.

measure_names (`List[str]`) –

The trec_eval measure names that will be computed. Measure names must be measure names supported by the official trec_eval measure. List of supported measures can be found in self.SUPPORTED_MEASURES. Note that: (1) We also accept the parametrized versions (e.g. "measure_name.k") of self.SUPPORTED_MEASURES measures. (2) We accept any measure that's in either "measure_name" or "measure_name.k" form, where measure_name is in pytrec_eval.supported_measures, but note that self.BINARY_MEASURES list must be modified to include any new binary measures.

correct_output (`str`) –

If the ADAPT_RANKING_BINARY mode is selected, the string that should be outputted if the model predicts that the object given in the instance can answer the question.

wrong_output (`str`) –

If the ADAPT_RANKING_BINARY mode is selected, the string that should be outputted if the model predicts that the object given in the instance can not answer the question.

rank (`Optional[int]`, default:`None`) –

The optional number of max top document rankings to keep for evaluation. If None, all the rankings are evaluated. If specified, only the documents that have a rank up to and including the specified rank are evaluated.

multiple_relevance_values (`bool`, default:`False`) –

Query relevance values can either be binary or take on multiple values, as explained below. This flag indicates whether the relevance values can take multiple values. (1) Binary relevance values: If the relevance values are binary, it means that all the matching relationships would get assigned a relevance value of 1, while the known non-matching relationships would get assigned a relevance value of 0. (2) Multiple relevance values: In the case of multiple relevance values, the value of 0 will be interpreted as non-matching relationship, but any other value would be interpreted as a matching relationship differing strengths.

| Parameters: |
| --- |

#### evaluate_references(adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Assign a score to the ranking of the references of an instance.

## reka_vibe_critique_metrics

### RekaVibeCritiqueMetric(num_respondents: int, max_tokens: int)

Critique evaluation for evaluating the correctness of generated response given the image and reference by Reka-vibe-eval.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## ruler_qa_metrics

### RulerQAMetric

Accuracy metric for Ruler QA Scenarios.

Adapted from: [https://github.com/NVIDIA/RULER/blob/1c45e5c60273e0ae9e3099137bf0eec6f0395f84/scripts/eval/synthetic/constants.py#L25](https://github.com/NVIDIA/RULER/blob/1c45e5c60273e0ae9e3099137bf0eec6f0395f84/scripts/eval/synthetic/constants.py#L25)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## safety_metrics

### SafetyBasicGenerationMetric()

Replacement for BasicGenerationMetric for HELM Safety. We call compute_request_state_metrics here because we can't use `BasicGenerationMetric` because we abuse "references" to store metadata rather than true metadata.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

### SafetyScoreMetric

Score metrics for HELM Safety.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## seahelm_metrics

### SEAHELMMachineTranslationMetric()

Machine Translation Metrics

This class computes the following standard machine translation metrics

1. chr_f_plus_plus (ChrF++)

@inproceedings{popovic-2015-chrf, title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation", author = "Popovi{'c}, Maja", editor = "Bojar, Ond{ {r}}ej and Chatterjee, Rajan and Federmann, Christian and Haddow, Barry and Hokamp, Chris and Huck, Matthias and Logacheva, Varvara and Pecina, Pavel", booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation", month = sep, year = "2015", address = "Lisbon, Portugal", publisher = "Association for Computational Linguistics", url = " [https://aclanthology.org/W15-3049](https://aclanthology.org/W15-3049)", doi = "10.18653/v1/W15-3049", pages = "392--395", github = " [https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu)", }

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

### SEAHELMQAMetric(language: str = 'en')

SEAHELM QA Metrics

This class computes the following standard SQuAD v1.1 metrics

1. squad_exact_match_score (SQuAD exact match score)
2. squad_f1_score (SQuAD macro-averaged F1 score)

@inproceedings{rajpurkar-etal-2016-squad, title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text", author = "Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy", editor = "Su, Jian and Duh, Kevin and Carreras, Xavier", booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing", month = nov, year = "2016", address = "Austin, Texas", publisher = "Association for Computational Linguistics", url = " [https://aclanthology.org/D16-1264](https://aclanthology.org/D16-1264)", doi = "10.18653/v1/D16-1264", pages = "2383--2392", }

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## spider_metrics

### SpiderMetric

Score metrics for Spider. Based on Bird-SQL.

## summarization_critique_metrics

### SummarizationCritiqueMetric(num_respondents: int)

Reimplementation of SummarizationMetric's evals using critique evaluation.

This is a demonstration of critique evaluation and is not intended for production use.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

Get critiques of a summary and compute metrics based on the critiques.

## summarization_metrics

### SummarizationMetric(task: str, language: str = 'en', device: str = 'cpu', bertscore_model: str = 'microsoft/deberta-large-mnli', rescale_with_baseline: bool = True, summac_new_line_split: bool = False)

Summarization Metrics

This class computes the following standard summarization metrics

1. Rouge (1,2,L)
2. Extractiveness (coverage, density, novel n-grams)
3. Compression
4. Faithfulness (SummaC)

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

## toxicity_metrics

### ToxicityMetric()

Defines metrics for toxicity.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]

We compute the same metrics from the RealToxicityPrompts paper: [https://arxiv.org/pdf/2009.11462.pdf](https://arxiv.org/pdf/2009.11462.pdf)

The authors used the following metrics to evaluate the language models with toxic and non-toxic prompts separately:

1. Expected maximum toxicity over k generations. We call this "expected_max_toxicity".
2. Empirical probability of generating a span with Toxicity >= 0.5 at least once over k generations. We call this "max_toxicity_probability".

We also compute the fraction of completions with Toxicity >= 0.5 ("toxic_frac") and count the number of completions the model generated ("num_completions").

## ultra_suite_asr_classification_metrics

### UltraSuiteASRMetric

Score metrics for UltraSuite ASR.

#### evaluate_instances(request_states: List[RequestState], eval_cache_path: str) -> List[Stat]

## unitxt_metrics

### UnitxtMetric(**kwargs)

## wildbench_metrics

### WildBenchScoreMetric

Score metrics for WildBench.

#### evaluate_generation(adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]