from typing import Optional, List
import re

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class RoboRewardBenchPreferenceRankingMetric(Metric):
    @staticmethod
    def extract_answer(raw_completion: str) -> Optional[str]:
        # Expected format: "ANSWER: a" or "ANSWER: b" or "ANSWER: tie"
        extracted_answer: str = raw_completion
        if "ANSWER:" in extracted_answer:
            extracted_answer = extracted_answer.split("ANSWER:")[1]
        elif "Answer:" in extracted_answer:
            extracted_answer = extracted_answer.split("Answer:")[1]

        extracted_answer = extracted_answer.strip()
        if extracted_answer.lower() in ["a", "b", "tie", "tied"]:
            return extracted_answer
        else:
            return None

    def __repr__(self):
        return "RoboRewardBenchPreferenceRankingMetric"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        instance = request_state.instance
        assert instance.references and len(instance.references) == 1
        correct_answer: str = instance.references[0].output.text.strip()

        if request_state.result is None or not request_state.result.completions:
            hlog(f"Missing result for instance {instance.id}")
            return []

        prediction: str = request_state.result.completions[0].text
        predicted_answer: Optional[str] = self.extract_answer(prediction)

        if predicted_answer is None:
            hlog(f"Could not extract answer for instance {instance.id}: {prediction}")
            score = 0.0
        else:
            if predicted_answer.lower() == correct_answer.lower():
                score = 1.0
            elif predicted_answer.lower() in ["tie", "tied"] and correct_answer.lower() in ["tie", "tied"]:
                score = 1.0
            else:
                score = 0.0

        return [Stat(MetricName("exact_match", split="test")).add(score)]


class RoboRewardBenchProgressPredictionMetric(Metric):
    @staticmethod
    def extract_answer(raw_completion: str) -> Optional[float]:
        """
        Extracts the first numeric score from model output.
        Supports:
          - 'ANSWER: 2. Explanation...'
          - 'ANSWER: 2'
          - '5. Explanation...' (number at start of output)
        """
        if not raw_completion:
            return None
        text = str(raw_completion).strip()

        # Case 1 & 2: explicit 'ANSWER: <number>' (case-insensitive), possibly followed by explanation
        m = re.search(r'(?i)\banswer\b\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            return float(m.group(1))

        # Case 3: output starts with a number like "5. ..." or "5 "
        m = re.search(r'^\s*(-?\d+(?:\.\d+)?)\b', text)
        if m:
            return float(m.group(1))

        return None

    def __repr__(self):
        return "RoboRewardBenchProgressPredictionMetric"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        instance = request_state.instance
        # ground truth is a single float in references[0].output.text
        correct_answer = float(instance.references[0].output.text.strip())

        if request_state.result is None or not request_state.result.completions:
            hlog(f"Missing result for instance {instance.id}")
            return []

        prediction = request_state.result.completions[0].text
        predicted_score = self.extract_answer(prediction)

        if predicted_score is None:
            hlog(f"Could not extract answer for instance {instance.id}: {prediction}")
            abs_err = 4.0
            sq_err = 16.0
        else:
            # compute absolute error and squared error
            abs_err = abs(predicted_score - correct_answer)
            sq_err = (predicted_score - correct_answer) ** 2

        return [
            Stat(MetricName("abs_error", split="test")).add(abs_err),
            Stat(MetricName("squared_error", split="test")).add(sq_err),
            Stat(MetricName("exact_match", split="test")).add(1.0 if predicted_score == correct_answer else 0.0),
        ]
