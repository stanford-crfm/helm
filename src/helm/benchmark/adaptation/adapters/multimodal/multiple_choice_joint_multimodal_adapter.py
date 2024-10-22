from abc import ABC
from typing import Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.multimodal.in_context_learning_multimodal_adapter import (
    InContextLearningMultimodalAdapter,
)
from helm.benchmark.adaptation.adapters.multimodal.multimodal_prompt import MultimodalPrompt


class MultipleChoiceJointMultimodalAdapter(InContextLearningMultimodalAdapter, ABC):
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`. This `Adapter` has additional logic to support in-context
    learning for multimodal models.
    """

    @staticmethod
    def get_reference_prefix(prefix: str, i: int) -> str:
        """
        Example: prefix = "\nA. ", i = 2, return "\nC. "
        """
        return prefix.replace("A", chr(ord("A") + i))

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        prompt: MultimodalPrompt = self.construct_prompt(
            training_instances, eval_instance, include_output=False, reference_index=None
        )
        output_mapping: Dict[str, str] = dict(
            (self.get_reference_prefix("A", reference_index), reference.output.text)
            for reference_index, reference in enumerate(eval_instance.references)
        )
        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            multimodal_prompt=prompt.multimedia_object,
            num_completions=self.adapter_spec.num_outputs,
            temperature=self.adapter_spec.temperature,
            max_tokens=self.adapter_spec.max_tokens,
            stop_sequences=[],
            random=self.adapter_spec.random,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=train_trial_index,
            output_mapping=output_mapping,
            request=request,
            result=None,
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=False,
        )
        return [request_state]

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

        # Include the references
        delimiter: str = ", "
        no_correct_references: str = "n/a"
        output: str = no_correct_references
        for reference_index, reference in enumerate(instance.references):
            prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)

            if reference.output.multimedia_content is not None:
                reference_output_content: MultimediaObject = reference.output.multimedia_content
                reference_output_content = reference_output_content.add_textual_prefix(prefix)
                reference_output_content = reference_output_content.add_textual_suffix(
                    self.adapter_spec.reference_suffix
                )
                result = result.combine(reference_output_content)
            else:
                result = result.add_textual_suffix(prefix + reference.output.text + self.adapter_spec.reference_suffix)

            if reference.is_correct:
                if output == no_correct_references:
                    output = self.get_reference_prefix("A", reference_index)
                elif self.adapter_spec.multi_label:
                    output += delimiter
                    output += self.get_reference_prefix("A", reference_index)

        if include_output:
            output_content: MultimediaObject = MultimediaObject([MediaObject(text=output, content_type="text/plain")])
            output_content = output_content.add_textual_prefix(self.adapter_spec.output_prefix)
            output_content = output_content.add_textual_suffix(self.adapter_spec.output_suffix)
            result = result.combine(output_content)
        else:
            result = result.add_textual_suffix(self.adapter_spec.output_prefix.rstrip())

        return result
