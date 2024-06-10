import math
import json
from typing import List, Union

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.fin_qa_metrics_helper import (  # type: ignore
    equal_program,
    eval_program,
    program_tokenization,
)


def _get_program_accuracy(reference_program: List[str], generated_program: List[str]) -> float:
    return 1.0 if equal_program(reference_program, generated_program) else 0.0


def _get_execution_accuracy(reference_execution: str, generated_program: List[str], table: List[List[str]]) -> float:
    invalid_flag: int
    generated_result: Union[str, float]
    invalid_flag, generated_result = eval_program(generated_program, table)
    if invalid_flag:
        return 0.0
    if reference_execution == "yes" or reference_execution == "no":
        return 1.0 if reference_execution == generated_result else 0
    else:
        if not isinstance(generated_result, float):
            return 0.0
        return 1.0 if math.isclose(float(reference_execution), generated_result) else 0


class FinQAMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert len(request_state.instance.references) == 3
        reference_text = request_state.instance.references[0].output.text
        reference_program = program_tokenization(reference_text)
        reference_execution = request_state.instance.references[1].output.text
        table: List[List[str]] = json.loads(request_state.instance.references[2].output.text)

        assert request_state.result
        assert len(request_state.result.completions) == 1
        generated_text = request_state.result.completions[0].text.strip()
        generated_program = program_tokenization(generated_text)

        return [
            Stat(MetricName("program_accuracy")).add(_get_program_accuracy(reference_program, generated_program)),
            Stat(MetricName("execution_accuracy")).add(
                _get_execution_accuracy(reference_execution, generated_program, table)
            ),
        ]
