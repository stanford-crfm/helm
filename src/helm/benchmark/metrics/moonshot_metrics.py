import asyncio
import importlib
import inspect
import numbers
import os
import sys

from typing import Any, Dict, List

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import MetricInterface, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT
from helm.common.general import ensure_file_downloaded


_METRICS_BASE_URL = "https://raw.githubusercontent.com/aiverify-foundation/moonshot-data/main/metrics"


def _ensure_metric_downloaded(metric_id, eval_cache_path: str) -> str:
    file_name = f"{metric_id}.py"
    source_url = f"{_METRICS_BASE_URL}/{file_name}"
    file_path = os.path.join(eval_cache_path, file_name)
    ensure_file_downloaded(source_url=source_url, target_path=file_path)
    return file_path


def _import_moonshot_metric_module(metric_id: str, eval_cache_path: str):
    file_path = _ensure_metric_downloaded(metric_id, eval_cache_path)
    module_name = f"moonshot.data.metrics.{metric_id}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_moonshot_metric_class(metric_id: str, eval_cache_path: str):
    module = _import_moonshot_metric_module(metric_id, eval_cache_path)
    for attr_name in dir(module):
        module_member = getattr(module, attr_name)
        if inspect.isclass(module_member) and module_member.__name__.lower() == metric_id:
            return module_member
    raise ValueError("Could not find thing")


class MoonshotMetric(MetricInterface):
    def __init__(self, metric_id: str):
        super().__init__()
        self.metric_id = metric_id
        
    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        moonshot_metric = _get_moonshot_metric_class(self.metric_id, eval_cache_path)()
        prompts: List[str] = []
        predicted_results: List[str] = []
        targets: List[str] = []
        for request_state in scenario_state.request_states:
            prompts.append(request_state.request.prompt)
            assert len(request_state.instance.references) == 1
            assert len(request_state.instance.references[0].tags) == 1
            assert request_state.instance.references[0].tags[0] == CORRECT_TAG
            targets.append(request_state.instance.references[0].output.text)
            assert request_state.result
            for generated_output in request_state.result.completions:
                predicted_results.append(generated_output.text)
        results = asyncio.run(moonshot_metric.get_results(prompts, predicted_results, targets))
        aggregated_stats: List[Stat] = []
        for result_name, result_value in results.items():
            if isinstance(result_value, numbers.Number):
                aggregated_stats.append(Stat(MetricName(name=result_name, split=TEST_SPLIT)).add(result_value))
        return MetricResult(aggregated_stats=aggregated_stats, per_instance_stats=[])
