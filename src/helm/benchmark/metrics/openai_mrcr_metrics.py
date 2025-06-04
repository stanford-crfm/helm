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
    """
    This metric focuses on structured reasoning and the accuracy of extracted answers.
    It compares model outputs against correct answers provided in a multiple-choice
    format and returns a score indicating the correctness of the generated response.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate the generated output for chain-of-thought reasoning accuracy.

        The method extracts the model's output, determines the correct answer
        from the provided references, and compares the two to compute a binary score.

        Args:
            adapter_spec (AdapterSpec): Specification of the adapter used for the evaluation.
            request_state (RequestState): The state of the current request, including
                                          the input instance, output results, and references.
            metric_service (MetricService): A service used to compute metrics if needed.
            eval_cache_path (str): Path to the evaluation cache for storing or retrieving data.

        Returns:
            List[Stat]: A list containing a single `Stat` object with the correctness
                        score (1 for correct, 0 for incorrect) under the metric
                        name "chain_of_thought_correct".
        """
        # Assert that completions exist if the result is not None
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
