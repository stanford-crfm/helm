import re
from typing import List

from math_verify import parse, verify

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Reference


def _extract_boxed(text: str) -> str:
    """Return the last \\boxed{} content if present, otherwise the stripped text."""
    matches = re.findall(r"\\boxed\\s*{([^}]*)}", text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _get_correct_reference(references: List[Reference]) -> Reference:
    for reference in references:
        if reference.is_correct:
            return reference
    return references[0]


class MathVerifyMetric(Metric):
    """Accuracy metric that uses math_verify to compare model outputs with gold answers."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None and request_state.result.completions, "Missing completions."

        output_text = request_state.result.completions[0].text
        gold_reference = _get_correct_reference(request_state.instance.references)
        gold_text = gold_reference.output.text

        predicted_expr = _extract_boxed(output_text)
        gold_expr = _extract_boxed(gold_text)

        is_correct = False
        try:
            gold_parsed = parse(gold_expr)
            predicted_parsed = parse(predicted_expr)
            is_correct = bool(verify(gold_parsed, predicted_parsed))
        except Exception:
            is_correct = gold_expr.strip() == predicted_expr.strip()

        return [Stat(MetricName("math_accuracy")).add(int(is_correct))]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="math_accuracy",
                display_name="Math accuracy",
                short_display_name="Acc",
                description="Math equivalence accuracy computed with math_verify.",
                lower_is_better=False,
                group="accuracy",
            ),
        ]

