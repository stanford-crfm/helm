from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class RegressionMetric(Metric):
    """Score metrics for regression tasks."""

    def __init__(self, default: float = 0.0):
        super().__init__()
        self.default = default

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        try:
            reference_value: float = float(request_state.instance.references[0].output.text)
        except (IndexError, ValueError):
            raise ValueError("Reference value is missing or not a valid float.")

        completions: List[str] = [c.text for c in request_state.result.completions]
        list_completion_values: List[float] = []
        for completion in completions:
            try:
                completion_value: float = float(completion)
            except ValueError:
                continue

            list_completion_values.append(completion_value)

        if not list_completion_values:
            list_completion_values = [self.default]

        mean_completion_value = sum(list_completion_values) / len(list_completion_values)

        result = {
            "mean_absolute_error": mean_absolute_error([reference_value], [mean_completion_value]),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(
                [reference_value], [mean_completion_value]
            ),
            "mean_squared_error": mean_squared_error([reference_value], [mean_completion_value]),
            "root_mean_squared_error": root_mean_squared_error([reference_value], [mean_completion_value]),
            "r2_score": r2_score([reference_value], [mean_completion_value]),
        }

        return [
            Stat(MetricName("mean_absolute_error")).add(result["mean_absolute_error"]),
            Stat(MetricName("mean_absolute_percentage_error")).add(result["mean_absolute_percentage_error"]),
            Stat(MetricName("mean_squared_error")).add(result["mean_squared_error"]),
            Stat(MetricName("root_mean_squared_error")).add(result["root_mean_squared_error"]),
            Stat(MetricName("r2_score")).add(result["r2_score"]),
        ]
