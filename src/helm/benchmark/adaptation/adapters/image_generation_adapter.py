from typing import List, Optional

from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from .generation_adapter import GenerationAdapter


class ImageGenerationAdapter(GenerationAdapter):
    """
    For image generation, the text-to-image model will generate the output for prompt:
        <input>
    """

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        prompt: Prompt = self.construct_prompt(
            training_instances, eval_instance, include_output=False, reference_index=None
        )
        request = Request(
            model=self.adapter_spec.model,
            prompt=prompt.text,
            num_completions=self.adapter_spec.num_outputs,
            image_generation_parameters=self.adapter_spec.image_generation_parameters,
            max_tokens=0,
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
            prompt_truncated=prompt.truncated,
        )
        return [request_state]

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """Returns a single example of the text-to-image prompt"""
        return instance.input.text
