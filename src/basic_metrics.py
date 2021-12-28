from typing import List

from adapter import RequestState
from statistic import Stat
from metric import Metric


class BasicMetric(Metric):
    """Produce basic metrics that just look at probabilities."""

    def evaluate_generation(self, request_state: RequestState) -> List[Stat]:
        # TODO: make this more robust
        assert request_state.result is not None
        prediction = request_state.result.completions[0].text

        gold_reference = request_state.instance.first_correct_reference
        assert gold_reference is not None
        gold = gold_reference.output

        return [
            Stat("exact_match").add(prediction == gold),
            # Highest probability of the highest correct reference
            # Stat('generation_ranking'),
        ]

    def evaluate_references(self, reference_request_states: List[RequestState]) -> List[Stat]:
        # for reference in reference_request_states:
        # Highest ranking of any correct answer
        # TODO
        return []
