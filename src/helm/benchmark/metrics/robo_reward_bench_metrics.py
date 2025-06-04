from typing import Optional, List
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class RoboRewardBenchMetric(Metric):
    @staticmethod
    def extract_answer(raw_completion: str) -> Optional[str]:
        # Expected format: "ANSWER: a" or "ANSWER: b" or "ANSWER: tie"
        extracted_answer: str = raw_completion
        if "ANSWER:" in extracted_answer:
            extracted_answer = extracted_answer.split("ANSWER:")[1]

        extracted_answer = extracted_answer.strip()
        if extracted_answer.lower() in ["a", "b", "tie"]:
            return extracted_answer
        else:
            return None

    def __repr__(self):
        return "RoboRewardBenchMetric"

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
            score = 1.0 if predicted_answer.lower() == correct_answer.lower() else 0.0

        return [Stat(MetricName("exact_match", split="test")).add(score)]
