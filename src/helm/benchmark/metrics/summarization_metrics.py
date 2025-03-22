import json
import pandas
import numpy as np
import os
import pickle

import spacy
import spacy.cli
from typing import List, Dict, Optional
from collections import defaultdict

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.evaluate_reference_metrics import get_rouge_function
from helm.common.hierarchical_logger import hlog
from helm.common.general import ensure_file_downloaded
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.summac.model_summac import SummaCZS

try:
    from bert_score import BERTScorer  # type: ignore
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["summarization"])


QAFACTEVAL_URL: str = (
    "https://storage.googleapis.com/crfm-helm-public/source_datasets/metrics/summarization_metrics/qafacteval.pk"
)
HUMAN_EVAL_URL: str = (
    "https://storage.cloud.google.com/crfm-helm-public/source_datasets/metrics/summarization_metrics/{file_name}"
)


class SummarizationMetric(Metric):
    """Summarization Metrics

    This class computes the following standard summarization metrics

    1. Rouge (1,2,L)
    2. Extractiveness (coverage, density, novel n-grams)
    3. Compression
    4. Faithfulness (SummaC)
    """

    def __init__(
        self,
        task: str,
        device: str = "cpu",
        bertscore_model: str = "microsoft/deberta-large-mnli",
        rescale_with_baseline: bool = True,
    ):
        self.rouge_fns = {
            "rouge_1": get_rouge_function("rouge1"),
            "rouge_2": get_rouge_function("rouge2"),
            "rouge_l": get_rouge_function("rougeL"),
        }
        # Download en_core_web_sm before importing DataStatsMetric to
        # avoid triggering a bug in DataStatsMetric that raises
        # `NameError: name 'stderr' is not defined`
        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")

        try:
            from summ_eval.data_stats_metric import DataStatsMetric  # type: ignore
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["summarization"])

        self.data_stats_metric = DataStatsMetric()
        self.task: str = task
        self.qa_fact_eval: Optional[Dict] = None
        self.humaneval: Optional[Dict] = None

        if device == "cpu":
            self.compute_faithfulness = False
            self.compute_bertscore = False
        else:
            self.compute_bertscore = True
            self.bert_scorer = BERTScorer(
                model_type=bertscore_model, lang="en", rescale_with_baseline=rescale_with_baseline, device=device
            )
            # Need GPU for faithfulness metrics since they are model-based.
            self.compute_faithfulness = True
            self.summac = SummaCZS(granularity="sentence", model_name="vitc", imager_load_cache=False, device=device)

    def _load_qafacteval(self, eval_cache_path: str):
        target_path: str = os.path.join(eval_cache_path, "qafacteval.pk")
        ensure_file_downloaded(source_url=QAFACTEVAL_URL, target_path=target_path)

        with open(target_path, "rb") as fin:
            qafacteval_scores = pickle.load(fin)

        self.qa_fact_eval = qafacteval_scores[self.task]

    def _load_humaneval(self, eval_cache_path: str) -> Dict:
        """
        Load all human evaluation data cached on CodaLab into a single dictionary

        key: (metric_type: str, model_name: str, output_summary: str)
        value: corresponding score: float
        """
        if "cnndm" in self.task:
            dataset = "cnndm"
        elif "xsum" in self.task:
            dataset = "xsum"
        else:
            raise ValueError

        all_humaneval_scores = dict()
        for shots in [0, 5]:
            score_analyzer = SummarizationHumanEvalAnalyzer(dataset, eval_cache_path, shots=shots)
            for (model_name, input_id, output_text), score in score_analyzer.faithfulness_full.items():
                if isinstance(output_text, float):
                    output_text = ""
                all_humaneval_scores[("faithfulness", model_name, input_id, output_text)] = score
            for (model_name, input_id, output_text), score in score_analyzer.relevance_full.items():
                if isinstance(output_text, float):
                    output_text = ""
                all_humaneval_scores[("relevance", model_name, input_id, output_text)] = score
            for (model_name, input_id, output_text), score in score_analyzer.coherence_full.items():
                if isinstance(output_text, float):
                    output_text = ""
                all_humaneval_scores[("coherence", model_name, input_id, output_text)] = score
        return all_humaneval_scores

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        if self.compute_faithfulness:
            # When running with a GPU and parallelism > 1, errors with "...in layer_norm
            # return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
            # RuntimeError: expected scalar type Float but found Half".
            hlog(
                f"Setting parallelism from {parallelism} to 1, since "
                f"evaluating faithfulness with parallelism > 1 errors."
            )
            parallelism = 1

        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_rouge(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for metric, metric_fn in self.rouge_fns.items():
            metrics[metric] = np.max([metric_fn(ref, pred) for ref in refs])

        return metrics

    def _compute_data_stats(self, inp: str, pred: str) -> Dict[str, float]:
        stats = self.data_stats_metric.evaluate_example(pred, inp)
        return {
            "summarization_coverage": stats["coverage"],
            "summarization_density": stats["density"],
            "summarization_compression": stats["compression"],
        }

    def _compute_faithfulness_scores(self, inp: str, pred: str) -> Dict[str, float]:
        return {"summac": self.summac.score_one(inp, pred)["score"]}

    def _compute_bert_score(self, refs: List[str], pred: str) -> Dict[str, float]:
        p, r, f = self.bert_scorer.score([pred], [refs])
        return {"BERTScore-P": p[0].item(), "BERTScore-R": r[0].item(), "BERTScore-F": f[0].item()}

    def _remove_braces(self, text: str) -> str:
        if text.startswith("{"):
            text = text[1:]
        if text.endswith("}"):
            text = text[:-1]
        return text

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        refs: List[str] = [self._remove_braces(ref.output.text) for ref in request_state.instance.references]
        inp: str = self._remove_braces(request_state.instance.input.text)

        assert request_state.result is not None
        pred: str = self._remove_braces(request_state.result.completions[0].text.strip())

        result: List[Stat] = []

        try:
            if self.humaneval is None:
                self.humaneval = self._load_humaneval(eval_cache_path)

            # get human evaluation scores if they exist
            deployment = adapter_spec.model_deployment.replace("/", "_")
            for metric_name in ["faithfulness", "relevance", "coherence"]:
                val = self.humaneval[(metric_name, deployment, request_state.instance.id, pred)]
                result.append(Stat(MetricName(f"HumanEval-{metric_name}")).add(float(val)))
        except KeyError:
            pass
        except ValueError:
            pass  # HumanEval not available for this task

        try:
            # get qafacteval scores if they exist
            if self.qa_fact_eval is None:
                self._load_qafacteval(eval_cache_path)
            assert self.qa_fact_eval is not None
            deployment = adapter_spec.model_deployment.replace("/", "_")
            val = self.qa_fact_eval[deployment][(request_state.instance.id, pred)]
            result.append(Stat(MetricName("QAFactEval")).add(float(val)))
        except KeyError:
            pass
        except ValueError:
            pass  # QAFactEval not available for this task

        # Compute rouge metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_rouge(refs, pred).items()])

        # Compute data stats
        result.extend(
            [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_data_stats(inp, pred).items()]
        )

        # Compute faithfulness metric(s)
        if self.compute_faithfulness:
            result.extend(
                [
                    Stat(MetricName(name)).add(float(val))
                    for name, val in self._compute_faithfulness_scores(inp, pred).items()
                ]
            )

        # Compute BERTScore
        if self.compute_bertscore:
            result.extend(
                [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_bert_score(refs, pred).items()]
            )

        return result


def _paired_bootstrap_test(treatment_list: list, control_list: list, nboot: int = 10000):
    """
    Computes paired bootstrap test for the Hypothesis: treament > control

    Args:
        treatment: list of float, representing results of treament (better model results)
        control: list of float, representing results of control (worse model results)
        nboot: int, number of bootstraps to perform
    """
    treatment = np.array(treatment_list)
    control = np.array(control_list)
    delta = treatment.mean() - control.mean()
    sample_idx = np.random.choice(np.arange(len(treatment)), size=(nboot, len(treatment)))
    boot_treatment = treatment[sample_idx]
    boot_control = control[sample_idx]
    diff = boot_treatment.mean(axis=1) - boot_control.mean(axis=1)
    return (diff > 2 * delta).mean()


class SummarizationHumanEvalAnalyzer:
    """
    Analyzes the human evaluation data of on summarization datasets

    1. loads human evaluation data from CodaLab
    2. averages and report {faithfulness, relevance, coherence} scores
    3. compute paired bootstrap test for all pairwise model comparison
    """

    def __init__(self, dataset: str, eval_download_path: str, shots: int):
        self.dataset = dataset
        self.eval_download_path = eval_download_path
        self.shots = shots
        os.makedirs(eval_download_path, exist_ok=True)
        self.load_humaneval_data()

    def load_humaneval_data(self):
        filename = f"{self.dataset}_{self.shots}shots.csv"

        tasks_by_id = defaultdict(list)

        download_filename = HUMAN_EVAL_URL.format(file_name=filename)
        filename = os.path.join(self.eval_download_path, filename)
        ensure_file_downloaded(source_url=download_filename, target_path=filename)
        mturk_data = pandas.read_csv(filename)
        for i, row in mturk_data.iterrows():
            tasks_by_id[row.HITId].append(row)

        self.faithfulness = defaultdict(list)
        self.faithfulness_full = dict()
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                # Faithfulness is evaluated as a binary choice
                # False -> not Faithful
                # True -> Faithful
                scores.append(1 if task["Answer.consistency.consistent"] else 0)
            self.faithfulness_full[(task["Input.model_name"], task["Input.id"], task["Input.output_text"])] = np.mean(
                scores
            )
            self.faithfulness[task["Input.model_name"]].append(np.mean(scores))

        self.coherence = defaultdict(list)
        self.coherence_full = dict()
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                # Coherence is evaluated on a 1 to 5 Likert scale.
                # 1 -> least coherent
                # 5 -> most coherent
                for i in range(1, 6):
                    if task[f"Answer.coherence.cohere_{i}"]:
                        scores.append(i)
                        break
            self.coherence_full[(task["Input.model_name"], task["Input.id"], task["Input.output_text"])] = np.mean(
                scores
            )
            self.coherence[task["Input.model_name"]].append(np.mean(scores))

        self.relevance = defaultdict(list)
        self.relevance_full = dict()
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                # Relevance is evaluated on a 1 to 5 Likert scale.
                # 1 -> least relevant
                # 5 -> most relevant
                for i in range(1, 6):
                    if task[f"Answer.relevance.rel_{i}"]:
                        scores.append(i)
                        break
            self.relevance_full[(task["Input.model_name"], task["Input.id"], task["Input.output_text"])] = np.mean(
                scores
            )
            self.relevance[task["Input.model_name"]].append(np.mean(scores))

    def _compute_average(self, scores: dict):
        """
        Computes average for each entry in a {model_name: score_list} dict
        """
        return [(x, np.mean(y)) for x, y in scores.items()]

    def print_summary(self):
        assert self.faithfulness
        assert self.coherence
        assert self.relevance

        print("FAITHFULNESS")
        for model, score in self._compute_average(self.faithfulness):
            print(f"{model:40}: {score:.4f}")
        print("=" * 40)

        print("RELEVANCE")
        for model, score in self._compute_average(self.relevance):
            print(f"{model:40}: {score:.4f}")
        print("=" * 40)

        print("COHERENCE")
        for model, score in self._compute_average(self.relevance):
            print(f"{model:40}: {score:.4f}")
        print("=" * 40)

    def dump_test_result(self, output_file_path: str):
        """
        Dumps pair-wise model comparison results (based on paired bootstrap test) into a json file.

        Output format:
        {
            "faithfulness":[
                {"model1": ..., "model2": ..., "p value: ...}
            ]
            "relevance": ...,
            "coherence": ...
        }

        Args:
            output_file_path: str, path to the output json file
        """
        assert self.faithfulness
        assert self.coherence
        assert self.relevance

        output_pvalues = defaultdict(list)

        avg_faithful_scores = self._compute_average(self.faithfulness)
        sorted_models, _ = zip(*sorted(avg_faithful_scores, key=lambda x: x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i + 1 :]:
                p_value = _paired_bootstrap_test(self.faithfulness[best_model], self.faithfulness[other_model])
                output_pvalues["faithfulness"].append({"model1": best_model, "model2": other_model, "p value": p_value})

        avg_relevance_scores = self._compute_average(self.relevance)
        sorted_models, _ = zip(*sorted(avg_relevance_scores, key=lambda x: x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i + 1 :]:
                p_value = _paired_bootstrap_test(self.relevance[best_model], self.relevance[other_model])
                output_pvalues["relevance"].append({"model1": best_model, "model2": other_model, "p value": p_value})

        avg_coherence_scores = self._compute_average(self.coherence)
        sorted_models, _ = zip(*sorted(avg_coherence_scores, key=lambda x: x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i + 1 :]:
                p_value = _paired_bootstrap_test(self.coherence[best_model], self.coherence[other_model])
                output_pvalues["coherence"].append({"model1": best_model, "model2": other_model, "p value": p_value})

        with open(output_file_path, "w") as f:
            json.dump(dict(output_pvalues), f)
