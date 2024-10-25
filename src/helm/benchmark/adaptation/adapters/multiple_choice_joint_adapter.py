from typing import List, Dict, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter


class MultipleChoiceJointAdapter(InContextLearningAdapter):
    """
    Each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

    We can define a label (e.g., letter) for each reference:

        <instructions>

        <input>                  # train
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer: C

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:

    In general, each example is:

        <input_prefix><input><reference_prefixes[index]><reference><output_prefix><output>
    """

    @staticmethod
    def get_prefix_char(prefix: str) -> str:
        return [char for char in prefix if char.isalnum()][0]

    @staticmethod
    def get_reference_prefix(prefix: str, i: int) -> str:
        """
        Example: prefix = "\nA. ", i = 2, return "\nC. "
        """
        prefix_char = MultipleChoiceJointAdapter.get_prefix_char(prefix)
        return prefix.replace(prefix_char, chr(ord(prefix_char) + i))

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        prefix_char = MultipleChoiceJointAdapter.get_prefix_char(self.adapter_spec.reference_prefix)
        prompt = self.construct_prompt(training_instances, eval_instance, include_output=False, reference_index=None)
        output_mapping: Dict[str, str] = dict(
            (self.get_reference_prefix(prefix_char, reference_index), reference.output.text)
            for reference_index, reference in enumerate(eval_instance.references)
        )
        request = Request(
            model=self.adapter_spec.model,
            model_deployment=self.adapter_spec.model_deployment,
            prompt=prompt.text,
            num_completions=1,
            top_k_per_token=self.adapter_spec.num_outputs,
            temperature=self.adapter_spec.temperature,  # usually this is 0
            max_tokens=self.adapter_spec.max_tokens,  # usually this is 1
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
            prompt_truncated=prompt.truncated,
        )
        return [request_state]

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """Return a list of lines corresponding to this example (part of the prompt)."""
        # Input
        result: str = self.adapter_spec.input_prefix + instance.input.text + self.adapter_spec.input_suffix

        # Include the references
        delimiter = ", "
        no_correct_references = "n/a"
        prefix_char = MultipleChoiceJointAdapter.get_prefix_char(self.adapter_spec.reference_prefix)
        output = no_correct_references
        for reference_index, reference in enumerate(instance.references):
            prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
            result += prefix + reference.output.text + self.adapter_spec.reference_suffix
            if reference.is_correct:
                if output == no_correct_references:
                    output = self.get_reference_prefix(prefix_char, reference_index)
                elif self.adapter_spec.multi_label:
                    output += delimiter
                    output += self.get_reference_prefix(prefix_char, reference_index)

        if include_output:
            result += self.adapter_spec.output_prefix + output + self.adapter_spec.output_suffix
        else:
            result += self.adapter_spec.output_prefix.rstrip()

        return result
