from dataclasses import replace
from typing import List, Dict

from common.request import Token
from common.statistic import Stat, merge_stat
from .adapter import AdapterSpec, ScenarioState, RequestState
from .metric_service import MetricService
from .metric import Metric
from .metric_name import MetricName
from .commonsense_qa_scenario import CLM_CORRECT_TAG


class CommonSenseQAMetric(Metric):
    """
    Metrics for CommonsenseQA datasets using causal language modeling
    TODO: Make it available (multiple requests per instance + calibration)
    in Adapter beyond the commonsense scenarios.
    """

    def __init__(self, n_choice: int = 4):
        self.n_choice = n_choice
        self.n_request_per_instance = self.n_choice * 2  # each choice w/ context or w/o context

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> List[Stat]:
        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}  # MetricName -> Stat

        for train_trial_index in range(adapter_spec.num_train_trials):
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial
            if scenario_state.request_states is not None:
                for idx in range(0, len(scenario_state.request_states), self.n_request_per_instance):
                    request_states = scenario_state.request_states[idx : idx + self.n_request_per_instance]

                    instance_stats = self.evaluate_references(adapter_spec, request_states, metric_service)

                    for stat in instance_stats:
                        # All of the request states should be for the same split
                        splits = list(set(request_state.instance.split for request_state in request_states))
                        assert len(splits) == 1

                        stat = Stat(replace(stat.name, split=splits[0])).merge(stat)
                        merge_stat(trial_stats, stat)

            # We only take the mean value for each trial
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        return list(global_stats.values())

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the logprob and normalization factors for the first completion"""
        assert request_state.result is not None
        assert len(request_state.instance.references) == 1
        assert len(request_state.result.completions) == 1

        sequence = request_state.result.completions[0]
        reference = request_state.instance.references[0].output

        # Only compute language modeling metrics for the reference
        answer_length = 0
        answer_tokens: List[Token] = []
        for token in sequence.tokens[::-1]:
            if answer_length >= len(reference):
                break
            answer_tokens.insert(0, token)
            answer_length += len(token.text)
        assert "".join([token.text for token in answer_tokens]).lstrip() == reference

        logprob = sum(token.logprob for token in answer_tokens)
        num_tokens = len(answer_tokens)

        return [
            Stat(MetricName("logprob")).add(logprob),
            Stat(MetricName("num_tokens")).add(num_tokens),
        ]

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService
    ) -> List[Stat]:
        """Evaluate the references."""
        assert len(reference_request_states) == self.n_request_per_instance
        stats = [
            self.evaluate_generation(adapter_spec, request_state, metric_service)
            for request_state in reference_request_states
        ]
        choice_labels = [
            CLM_CORRECT_TAG in request_state.instance.references[0].tags
            for request_state in reference_request_states[::2]
        ]  # [::2] because each choice has two requests: one w/ context and one w/o context
        assert choice_labels.count(True) == 1
        answer = choice_labels.index(True)

        # Original: sum of token logprob in answer given context / num of tokens in answer
        original_logprobs = [stats[i][0].mean / stats[i][1].mean for i in range(0, self.n_request_per_instance, 2)]
        # Calibration: sum of token logprob in answer given context - sum of token logprob in answer without context
        calibrated_logprobs = [
            stats[i][0].mean - stats[i + 1][0].mean for i in range(0, self.n_request_per_instance, 2)
        ]
        return [
            Stat(MetricName("original_acc")).add(float(max(original_logprobs) == original_logprobs[answer])),
            Stat(MetricName("calibrated_acc")).add(float(max(calibrated_logprobs) == calibrated_logprobs[answer])),
        ]
