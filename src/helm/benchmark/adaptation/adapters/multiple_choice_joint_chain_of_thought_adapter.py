from typing import Optional

from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.adapters.multiple_choice_joint_adapter import MultipleChoiceJointAdapter


class MultipleChoiceJointChainOfThoughtAdapter(MultipleChoiceJointAdapter):
    """
    Each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

        <instance_chain_of_thought>

    We can define a label (e.g., letter) for each reference:

        <global_prefix>
        <instructions>
        <input_prefix>
        <input>                  # train
        <input_suffix>
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        <output_prefix>
        <chain_of_thought_prefix>
        <instance_chain_of_thought>
        <chain_of_thought_suffix>
        <output>
        <output_suffix>

        <input_prefix>
        <input>                  # test
        <input_suffix>
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        <output_prefix>
        <chain_of_thought_prefix>
        <instance_chain_of_thought>
        <chain_of_thought_suffix>
        <output>
        <output_suffix>
        <global_suffix>

    In general, each example is:

        <input_prefix><input><input_suffix><reference_prefixes[index]><reference> \
        <output_prefix><chain_of_thought_prefix><chain_of_thought><chain_of_thought_suffix><output><output_suffix>
    """

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """Return a list of lines corresponding to this example (part of the prompt)."""
        # Input
        result: str = self.adapter_spec.input_prefix + instance.input.text + self.adapter_spec.input_suffix

        # Include the references
        delimiter = ", "
        no_correct_references = "n/a"
        output = no_correct_references
        for reference_index, reference in enumerate(instance.references):
            prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
            result += prefix + reference.output.text + self.adapter_spec.reference_suffix
            if reference.is_correct:
                if output == no_correct_references:
                    output = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
                elif self.adapter_spec.multi_label:
                    output += delimiter
                    output += self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)

        if include_output:
            chain_of_thought = instance.extra_data.get("chain_of_thought", "") if instance.extra_data else ""
            chain_of_thought_block = (
                self.adapter_spec.chain_of_thought_prefix + chain_of_thought + self.adapter_spec.chain_of_thought_suffix
            )
            result += (
                self.adapter_spec.output_prefix + chain_of_thought_block + output + self.adapter_spec.output_suffix
            )
        else:
            result += self.adapter_spec.output_prefix.rstrip()

        return result
