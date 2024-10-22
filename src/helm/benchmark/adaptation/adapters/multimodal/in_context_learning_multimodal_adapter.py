from abc import ABC
from dataclasses import replace
from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter
from helm.benchmark.adaptation.adapters.multimodal.multimodal_prompt import MultimodalPrompt


class InContextLearningMultimodalAdapter(InContextLearningAdapter, ABC):
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`. This `Adapter` has additional logic to support in-context
    learning for multimodal models.
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

    def construct_prompt(
        self,
        train_instances: List[Instance],
        eval_instance: Instance,
        include_output: bool,
        reference_index: Optional[int],
    ):
        """
        Returns a prompt given:
        - the `self.adapter_spec.instructions`
        - the `train_instances` (in-context multimodal training examples)
        - the multimodal input part of the `eval_instance`
        - the `reference` if `include_output` is true (if reference_index is not None, the reference
        at the given index; otherwise, the first correct reference)

        Fits the prompt within the context window by removing in-context training examples.
        """
        # Text for in-context training instances
        train_instance_blocks: List[MultimediaObject] = [
            self.construct_example_multimodal_prompt(inst, include_output=True, reference_index=None)
            for inst in train_instances
        ]

        # Eval example text
        eval_instance_block: MultimediaObject = self.construct_example_multimodal_prompt(
            eval_instance, include_output=include_output, reference_index=reference_index
        )

        # Prompt
        prompt = MultimodalPrompt(
            global_prefix=self.adapter_spec.global_prefix,
            global_suffix=self.adapter_spec.global_suffix,
            instructions=self.adapter_spec.instructions,
            train_instance_blocks=train_instance_blocks,
            eval_instance_block=eval_instance_block,
            instance_prefix=self.adapter_spec.instance_prefix,
        )

        # Ensure prompt fits within the context window
        prompt = self._fit_multimodal_prompt(prompt)
        return prompt

    def _fit_multimodal_prompt(self, prompt: MultimodalPrompt) -> MultimodalPrompt:
        """
        The prompt consists of instructions, training instances, and the evaluation input.
        We fit the prompt within the max content length by removing training instances
        based solely on the text length of the prompt (i.e., we do not consider the non-textual
        modalities).
        Return the multimodal prompt that fits.
        """
        orig_train_instances_count: int = prompt.num_train_instances
        while prompt.num_train_instances > 0:
            if self.window_service.fits_within_context_window(
                text=prompt.multimedia_object.text,
                expected_completion_token_length=self.adapter_spec.max_tokens,
            ):
                removed_train_instances_count: int = orig_train_instances_count - prompt.num_train_instances
                if removed_train_instances_count > 0:
                    hlog(
                        f"The original constructed prompt exceeded the max context length. "
                        f"Removed {removed_train_instances_count} in-context examples to fit "
                        f"it within the context window."
                    )
                return prompt

            # Remove the last training example
            prompt = replace(
                prompt, train_instance_blocks=prompt.train_instance_blocks[: len(prompt.train_instance_blocks) - 1]
            )
        return prompt

    def construct_example_multimodal_prompt(
        self, instance: Instance, include_output: bool, reference_index: Optional[int]
    ) -> MultimediaObject:
        """
        Returns a single example of the prompt. `include_output` controls whether the gold output is included.
        """
        # Input
        assert instance.input.multimedia_content is not None
        result: MultimediaObject = instance.input.multimedia_content.add_textual_prefix(self.adapter_spec.input_prefix)
        result = result.add_textual_suffix(self.adapter_spec.input_suffix)

        if include_output:
            output: str = self.construct_output(instance, reference_index)
            output_content: MultimediaObject = MultimediaObject([MediaObject(text=output, content_type="text/plain")])
            output_content = output_content.add_textual_prefix(self.adapter_spec.output_prefix)
            output_content = output_content.add_textual_suffix(self.adapter_spec.output_suffix)
            result = result.combine(output_content)
        else:
            result = result.add_textual_suffix(self.adapter_spec.output_prefix.rstrip())

        return result
