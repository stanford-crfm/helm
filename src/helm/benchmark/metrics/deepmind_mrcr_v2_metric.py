from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.deepmind_mrcr_v2_utils import mrcr_v2_metric
from helm.benchmark.scenarios.scenario import CORRECT_TAG


class DeepMindMRCRV2Metric(Metric):
    METRIC_NAME = "deepmind_mrcr_v2_score"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result
        assert len(request_state.result.completions) == 1

        prediction = request_state.result.completions[0].text

        assert len(request_state.instance.references) == 1
        assert len(request_state.instance.references[0].tags) == 1
        assert request_state.instance.references[0].tags[0] == CORRECT_TAG

        target = request_state.instance.references[0].output.text

        return [Stat(MetricName(self.METRIC_NAME)).add(mrcr_v2_metric(prediction=prediction, target=target))]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name=self.METRIC_NAME,
                display_name="MRCR Score",
                description="MRCR Score",
                lower_is_better=False,
                group=None,
            )
        ]
