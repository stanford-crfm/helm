"""The deterministic zBench grader as HELM stats (see zbench_grader.py, vendored from zReports-Health/zbench)."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.zbench_grader import grade


class ZBenchMetric(Metric):
    """Stats: zbench_exact (Pass), zbench_credit (Pass 1, Half 0.5), zbench_abstain, zbench_fail, zbench_half,
    zbench_review."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        item = (request_state.instance.extra_data or {}).get("item")
        completions = request_state.result.completions if request_state.result else []
        text = completions[0].text if completions else ""
        verdict = grade(item, text)["verdict"] if item else "Error"
        return [
            Stat(MetricName("zbench_exact")).add(1.0 if verdict == "Pass" else 0.0),
            Stat(MetricName("zbench_credit")).add({"Pass": 1.0, "Half": 0.5}.get(verdict, 0.0)),
            Stat(MetricName("zbench_abstain")).add(1.0 if verdict == "Abstain" else 0.0),
            Stat(MetricName("zbench_fail")).add(1.0 if verdict == "Fail" else 0.0),
            Stat(MetricName("zbench_half")).add(1.0 if verdict == "Half" else 0.0),
            Stat(MetricName("zbench_review")).add(1.0 if verdict == "Review" else 0.0),
        ]
