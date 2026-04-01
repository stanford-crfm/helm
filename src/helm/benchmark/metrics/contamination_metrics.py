import re
from typing import List
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import MetricInterface, MetricMetadata, MetricResult, PerInstanceStats
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class TSGuessingMetric(MetricInterface):
    """
    Evaluation metric for the TS-Guessing contamination detection strategy.
    Applies rigorous text cleaning, including removing verbal prefixes (Answer/Resposta)
    and choice letters (A:, B-), before calculating Exact Match and ROUGE-L F1.
    """

    def __init__(self, language: str = "en"):
        self.language = language.lower().split("_")[0].split("-")[0]
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            hlog("Downloading NLTK 'punkt' package for TSGuessingMetric...")
            nltk.download("punkt", quiet=True)

    def _process_response(self, text: str) -> str:
        """Cleans the raw model response using enhanced TS-Guessing rules."""
        if not text:
            return ""
        
        processed_text = str(text).strip()
        
        # Layer 1: Remove verbal prefixes (Answer:, Resposta:, etc.)
        verbal_prefix_pattern = r"^(answer|resposta|the answer is|a resposta é|a resposta e)s?\s*[:\-]*\s*"
        processed_text = re.sub(verbal_prefix_pattern, "", processed_text, flags=re.IGNORECASE).strip()
            
        # Layer 2: Remove choice letter prefixes (A:, A., A -, B) etc.)
        choice_prefix_pattern = r"^[a-z][\s\.\:\)\-]+\s*"
        processed_text = re.sub(choice_prefix_pattern, "", processed_text, flags=re.IGNORECASE).strip()

        # Layer 3: Extract only the first sentence to ignore model chatter
        try:
            sentences = sent_tokenize(processed_text)
            if sentences:
                processed_text = sentences[0]
        except Exception as e:
            hlog(f"DEBUG: sent_tokenize failed: {e}")
            
        # Layer 4: Final sanitization (Remove [MASK], quotes, and brackets)
        processed_text = processed_text.replace("[MASK]", "").strip()
        
        if processed_text.startswith("[") and processed_text.endswith("]"):
            processed_text = processed_text[1:-1].strip()
            
        if (processed_text.startswith('"') and processed_text.endswith('"')) or \
           (processed_text.startswith("'") and processed_text.endswith("'")):
            processed_text = processed_text[1:-1].strip()
            
        return processed_text.lower()

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        """Evaluates instances and returns statistics aligned with HELM standards."""
        
        use_stemmer = (self.language == "en")
        scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=use_stemmer)

        em_scores = []
        rouge_scores = []
        per_instance_stats_list = []

        for request_state in scenario_state.request_states:
            golds = [ref for ref in request_state.instance.references if ref.is_correct]
            if not golds:
                continue
            
            gold_text = golds[0].output.text.lower().strip()

            if not request_state.result or not request_state.result.completions:
                clean_pred = ""
            else:
                raw_pred = request_state.result.completions[0].text
                clean_pred = self._process_response(raw_pred)

            is_match = 1.0 if clean_pred == gold_text else 0.0
            
            try:
                rouge_score = scorer.score(clean_pred, gold_text)["rougeLsum"].fmeasure if clean_pred and gold_text else 0.0
            except Exception:
                rouge_score = 0.0
                
            em_scores.append(is_match)
            rouge_scores.append(rouge_score)

            instance_stats = [
                Stat(MetricName("ts_guessing_exact_match")).add(is_match),
                Stat(MetricName("ts_guessing_rouge_l_f1")).add(rouge_score),
            ]
            per_instance_stats_list.append(
                PerInstanceStats(
                    instance_id=request_state.instance.id,
                    perturbation=request_state.instance.perturbation,
                    train_trial_index=request_state.train_trial_index,
                    stats=instance_stats,
                )
            )

        split = scenario_state.request_states[0].instance.split if scenario_state.request_states else "test"
        em_mean = sum(em_scores) / len(em_scores) if em_scores else 0.0
        rouge_mean = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        return MetricResult(
            [
                Stat(MetricName("ts_guessing_exact_match", split=split)).add(em_mean),
                Stat(MetricName("ts_guessing_rouge_l_f1", split=split)).add(rouge_mean)
            ], 
            per_instance_stats_list
        )

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="ts_guessing_exact_match",
                display_name="TS-Guessing EM",
                description="Exact Match after stripping prefixes, choice letters, and chatter.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="ts_guessing_rouge_l_f1",
                display_name="TS-Guessing ROUGE-L F1",
                description="ROUGE-L F1 after stripping prefixes, choice letters, and chatter.",
                lower_is_better=False,
            ),
        ]