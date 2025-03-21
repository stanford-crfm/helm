import re
from typing import Any, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG
from helm.common.hierarchical_logger import hlog


def _strip_string(str: str) -> Any:
    # from https://stackoverflow.com/a/4703508
    numeric_const_pattern = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
    match = re.search(numeric_const_pattern, str)
    if match:
        try:
            return float(str[match.start() : match.end()])
        except Exception:
            return None
    return None


def float_equiv(str1: str, str2: str, eps: float = 1e-6) -> float:
    """Check if two values have the same float value, up to a small tolerance.

    This is the implementation used in the IBM Enterprise Benchmark paper.

    Note: This is a "mostly-correct" equality function and does not handle some cases correctly:

    - If both values are non-floats, then it will always return 1.0,
      regardless of whether strings match.
    - If either of both values have different units (e.g. currency symbols,
      trailing "M" or "B", trailing %), the values will not be converted to the same
      units before comparison.
    """
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)

        if ss1 is None or ss2 is None:
            hlog("WARNING: float_equiv returning 1.0 because both values are non-floats")
            return 0.0
        return float(abs(ss1 - ss2) < eps)
    except Exception:
        return float(str1 == str2)


class ConvFinQACalcMetric(Metric):
    """Score metrics for AIRBench 2024."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        model_answer = request_state.result.completions[0].text

        assert len(request_state.instance.references) == 1
        assert len(request_state.instance.references[0].tags) == 1
        assert request_state.instance.references[0].tags[0] == CORRECT_TAG
        gold_answer = request_state.instance.references[0].output.text

        return [
            Stat(MetricName("float_equiv")).add(float_equiv(model_answer, gold_answer)),
        ]
