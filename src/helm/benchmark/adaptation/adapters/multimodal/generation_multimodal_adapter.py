from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.multimodal.in_context_learning_multimodal_adapter import (
    InContextLearningMultimodalAdapter,
)
from helm.benchmark.adaptation.adapters.multimodal.multimodal_prompt import MultimodalPrompt


class GenerationMultimodalAdapter(InContextLearningMultimodalAdapter):
    """
    For generation, the multimodal model will generate the output for the following:

        <instructions>

        Input: <multimodal input>                  # train
        Output: <reference>

        Input: <multimodal input>                  # test
        Output:
    """

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        prompt: MultimodalPrompt = self.construct_prompt(
            training_instances, eval_instance, include_output=False, reference_index=None
        )

        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            multimodal_prompt=prompt.multimedia_object,
            num_completions=self.adapter_spec.num_outputs,
            temperature=self.adapter_spec.temperature,
            max_tokens=self.adapter_spec.max_tokens,
            stop_sequences=self.adapter_spec.stop_sequences,
            random=self.adapter_spec.random,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=False,
        )
        return [request_state]
