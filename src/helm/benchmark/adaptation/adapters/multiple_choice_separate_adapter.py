from typing import List

from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter


class MultipleChoiceSeparateAdapter(InContextLearningAdapter):
    """
    Each answer choice sentence is scored independently, where the score is
    the sentence probability normalized by the sentence length.
    """

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        request_states: List[RequestState] = []

        for reference_index, reference in enumerate(eval_instance.references):
            prompt = self.construct_prompt(
                training_instances,
                eval_instance,
                include_output=True,
                reference_index=reference_index,
            )
            request_states.append(
                self.construct_request_state(prompt, reference_index, eval_instance, train_trial_index)
            )

        return request_states

    def construct_request_state(
        self,
        prompt: Prompt,
        reference_index: int,
        eval_instance: Instance,
        train_trial_index: int,
        request_mode: str = "original",
    ) -> RequestState:
        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            prompt=prompt.text,
            num_completions=1,
            temperature=0,
            max_tokens=0,
            stop_sequences=[],
            echo_prompt=True,
        )
        return RequestState(
            instance=eval_instance,
            reference_index=reference_index,
            request_mode=request_mode,
            train_trial_index=train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=prompt.truncated,
        )
