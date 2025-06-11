from difflib import SequenceMatcher
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


class OpenAIMRCRMetric(Metric):
    """Accuracy metric for OpenAI MRCR.

    The measured metric is the SequenceMatcher ratio as implemented in https://docs.python.org/3/library/difflib.html.
    The model must prepend an alphanumeric hash to the beginning of its answer. If this hash is not included,
    the match ratio is set to 0. If it is correctly included, the stripped sampled answer is compared to the
    stripped ground truth answer.

    Adapted from: https://huggingface.co/datasets/openai/mrcr/blob/204b0d4e8d9ca5c0a90bf942fdb2a5969094adc0/README.md
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result
        assert len(request_state.result.completions) == 1

        response_text = request_state.result.completions[0].text

        assert len(request_state.instance.references) == 1
        assert len(request_state.instance.references[0].tags) == 1
        assert request_state.instance.references[0].tags[0] == CORRECT_TAG

        gold_text = request_state.instance.references[0].output.text

        assert request_state.instance.extra_data
        assert "random_string_to_prepend" in request_state.instance.extra_data
        random_string_to_prepend = request_state.instance.extra_data["random_string_to_prepend"]

        score = 0.0
        if response_text.startswith(random_string_to_prepend):
            response_sequence = response_text.removeprefix(random_string_to_prepend)
            gold_sequence = gold_text.removeprefix(random_string_to_prepend)
            score = float(SequenceMatcher(None, response_sequence, gold_sequence).ratio())

        return [Stat(MetricName("openai_mrcr_accuracy")).add(score)]
