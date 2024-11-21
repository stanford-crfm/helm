from typing import List

from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter


class GenerationAdapter(InContextLearningAdapter):
    """
    Each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

    For generation, the language model will generate the output.

        <instructions>

        Input: <input>                  # train
        Output: <reference>

        Input: <input>                  # test
        Output:

    In general, each example is:

        <input_prefix><input><output_prefix><output>
    """

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        prompt: Prompt = self.construct_prompt(
            training_instances, eval_instance, include_output=False, reference_index=None
        )
        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            prompt=prompt.text,
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
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=prompt.truncated,
        )
        return [request_state]
