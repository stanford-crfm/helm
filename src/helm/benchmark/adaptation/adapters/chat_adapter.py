from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter


class ChatAdapter(InContextLearningAdapter):
    """
    Each `Instance` in a `Scenario` has a history of the format:

    [
        {"role": "user", "content": <user-content>},
        {"role": "assistant", "content": <assistant-content>},
        {"role": "user", "content": <user-content>},
        ...
    ]

    """

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        assert eval_instance.extra_data
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in eval_instance.extra_data["conversation"]
        ]
        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            messages=messages,
            num_completions=self.adapter_spec.num_outputs,
            temperature=self.adapter_spec.temperature,
            max_tokens=self.adapter_spec.max_tokens,
            stop_sequences=self.adapter_spec.stop_sequences,
            random=self.adapter_spec.random,
            image_generation_parameters=self.adapter_spec.image_generation_parameters,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_train_instances=0,
            prompt_truncated=False,
        )
        return [request_state]
