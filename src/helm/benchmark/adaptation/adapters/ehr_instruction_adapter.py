from typing import List, Optional

from helm.benchmark.adaptation.adapters.generation_adapter import GenerationAdapter
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import TRAIN_SPLIT, Instance
from helm.benchmark.window_services.window_service import EncodeResult
from helm.common.tokenization_request import TokenizationToken


# in the prompt templates for EHR instructions, this is the placeholder for the EHR part
# which we use to compute accurate tokenized sequence lengths
PROMPT_TEMPLATE_EHR_PLACEHOLDER = "{ehr}"


class EHRInstructionAdapter(GenerationAdapter):
    """
    Each instance consists of the following:

    EHRInstructionInput:
        question: the question to answer or instruction to follow
        ehr: the XML-tagged EHR to use as context to answer the question
        prompt_template: a string template for how to combine the question + ehr

    Reference output:
        text: the 'golden' clinician response to the question

    This Adapter combines the above into RequestStates with logic to truncate the EHR specifically
    to fit in the context window with enough room for the instruction/question and the specified
    amount of generated tokens.
    """

    def adapt(self, instances: List[Instance], parallelism: int) -> List[RequestState]:
        """
        Main adaptation method which takes all instances and turns them into `RequestState` objects.
        """
        # sanity check, since for now we assume that there are no training instances at all
        if any(instance.split == TRAIN_SPLIT for instance in instances):
            raise RuntimeError(f"Got train instances for {self.__class__.__name__} - expected only eval instances.")

        # use superclass implementation here
        return super().adapt(instances, parallelism)

    def construct_prompt(
        self,
        train_instances: List[Instance],  # unused
        eval_instance: Instance,
        include_output: bool,  # unused
        reference_index: Optional[int],  # unused
    ) -> Prompt:
        """
        Uses the instance to construct a prompt for a given eval instance.

        Parameters
        ----------
        eval_instance: Instance
            the instance we wish to use to construct the prompt
        """
        # start by simply getting the inputs
        question = eval_instance.input.text
        assert eval_instance.extra_data is not None
        ehr_text: str = eval_instance.extra_data["ehr"]
        prompt_template: str = eval_instance.extra_data["prompt_template"]
        full_prompt_text = prompt_template.format(question=question, ehr=ehr_text)

        # insert the question and see how many tokens we have so far
        prompt_with_instr_no_ehr_placeholder = prompt_template.format(question=question, ehr="")
        num_tokens_no_ehr = self.window_service.get_num_tokens(prompt_with_instr_no_ehr_placeholder)

        # number of tokens we can allow the EHR part to be
        target_ehr_num_tokens = (
            self.window_service.max_request_length - self.adapter_spec.max_tokens - num_tokens_no_ehr
        )

        # round-trip tokenization to get the correct token length we need
        # NOTE: we truncate from the left side so that the most recent pieces of the EHR are included in the context
        # as opposed to the canonical way of truncating from the right. This is done to match the MedAlign method.
        full_ehr_tokens: EncodeResult = self.window_service.encode(ehr_text, max_length=None, truncation=False)
        truncated_ehr_tokens: List[TokenizationToken] = full_ehr_tokens.tokens[-target_ehr_num_tokens:]
        ehr_truncated: str
        ehr_truncated = self.window_service.decode(truncated_ehr_tokens)

        # create the truncated prompt
        truncated_prompt_text = prompt_template.format(question=question, ehr=ehr_truncated)
        num_truncations = 1
        while (
            num_extra_tokens := self.adapter_spec.max_tokens
            + self.window_service.get_num_tokens(truncated_prompt_text)
            - self.window_service.max_request_length
        ) > 0:
            truncated_ehr_tokens = truncated_ehr_tokens[num_extra_tokens:]
            ehr_truncated = self.window_service.decode(truncated_ehr_tokens)
            truncated_prompt_text = prompt_template.format(question=question, ehr=ehr_truncated)
            num_truncations += 1

        # naively construct the full non-truncated prompt
        prompt = Prompt(
            global_prefix=self.adapter_spec.global_prefix,
            global_suffix=self.adapter_spec.global_suffix,
            instance_prefix=self.adapter_spec.instance_prefix,
            substitutions=self.adapter_spec.substitutions,
            instructions_block=self.adapter_spec.instructions,
            train_instance_blocks=[],
            eval_instance_block=full_prompt_text,
            truncated_text=truncated_prompt_text,
        )

        return prompt
