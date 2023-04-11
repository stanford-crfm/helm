from typing import List, Dict, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from .in_context_learning_adapter import InContextLearningAdapter


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

        <input_prefix><input><reference_prefixes[0]><reference><output_prefix><output>
    """

    @staticmethod
    def get_reference_prefix(prefix: str, i: int) -> str:
        """
        Example: prefix = "\nA. ", i = 2, return "\nC. "
        """
        return prefix.replace("A", chr(ord("A") + i))

    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        prompt = self.construct_prompt(self.train_instances, eval_instance, include_output=False, reference_index=None)
        output_mapping: Dict[str, str] = dict(
            (self.get_reference_prefix("A", reference_index), reference.output.text)
            for reference_index, reference in enumerate(eval_instance.references)
        )
        request = Request(
            model=self.adapter_spec.model,
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
            train_trial_index=self.train_trial_index,
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
        output = "n/a"
        for reference_index, reference in enumerate(instance.references):
            prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
            result += prefix + reference.output.text + self.adapter_spec.reference_suffix
            if reference.is_correct and output == "n/a":
                output = self.get_reference_prefix("A", reference_index)

        if include_output:
            result += self.adapter_spec.output_prefix + output + self.adapter_spec.output_suffix
        else:
            result += self.adapter_spec.output_prefix.rstrip()

        return result
