from typing import List

from helm.benchmark.adaptation.adapter_spec import TextToImageAdapterSpec
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import TextToImageRequest
from .generation_adapter import GenerationAdapter


class ImageGenerationAdapter(GenerationAdapter):
    """
    For image generation, the text-to-image model will generate the output for prompt:

        <instructions><input_prefix><input><output_prefix>
    """

    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        assert isinstance(self.adapter_spec, TextToImageAdapterSpec)
        prompt: Prompt = self.construct_prompt(
            self.train_instances, eval_instance, include_output=False, reference_index=None
        )
        request = TextToImageRequest(
            model=self.adapter_spec.model,
            prompt=prompt.text,
            num_completions=self.adapter_spec.num_outputs,
            guidance_scale=self.adapter_spec.guidance_scale,
            width=self.adapter_spec.width,
            height=self.adapter_spec.height,
            max_tokens=0,
            random=self.adapter_spec.random,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=self.train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=prompt.truncated,
        )
        return [request_state]
