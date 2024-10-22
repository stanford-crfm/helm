from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance, Reference, TRAIN_SPLIT, EVAL_SPLITS, CORRECT_TAG
from helm.common.request import Request
from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter


class BinaryRankingAdapter(InContextLearningAdapter):
    """
    Adaptation strategy for ranking tasks, reduced to binary ranking.

    For tasks that require ranking, such as information retrieval tasks,
    an instance corresponds to a single query for which documents will be
    ranked. Each reference of an instance corresponds to a single document.
    A single evaluation instance block then contains a query and a document,
    relevance of which with respect to the query will be judged by the
    model. That is, given:

        [input], [reference_1], ... [reference_k]

    We construct the following evaluation instance block:

        Passage: [reference_i]
        Query: [input]
        Does the passage answer the query?
        Answer: Yes | No

    A request consists of a single evaluation instance block and a
    number of training instance blocks. For each training instance selected,
    we add two training instance blocks, one containing a relevant passage
    and another containing a passage that's not relevant.
    """

    # TODO: It would be better if we read the following from the adapter spec.
    RANKING_CORRECT_LABEL: str = "Yes"
    RANKING_WRONG_LABEL: str = "No"

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        request_states = []
        request_mode = "original"
        for reference_index, reference in enumerate(eval_instance.references):
            prompt = self.construct_prompt(
                training_instances,
                eval_instance,
                include_output=False,
                reference_index=reference_index,
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
            )
            request_state = RequestState(
                instance=eval_instance,
                reference_index=reference_index,
                request_mode=request_mode,
                train_trial_index=train_trial_index,
                output_mapping=None,
                request=request,
                result=None,
                num_train_instances=prompt.num_train_instances,
                prompt_truncated=prompt.truncated,
            )
            request_states.append(request_state)
        return request_states

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """Return an example prompt for binary ranking tasks.

        In the binary ranking prompt specification, the model's task is to
        output RANKING_CORRECT_LABEL if the document included in the prompt
        contains an answer to the query. If the document included does not answer
        the query, the model is expected to output RANKING_WRONG_LABEL.

        Example prompt:
            Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20
            years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would
            be at room temperature. Hope this helps.
            Query: how many eye drops per ml
            Does the passage answer the query?
            Answer: Yes
        """
        if instance.split == TRAIN_SPLIT:
            reference_indices = list(range(len(instance.references)))
        elif instance.split in EVAL_SPLITS:
            assert reference_index is not None
            reference_indices = [reference_index]
        else:
            raise ValueError(f"Unknown split, expected one of: {[TRAIN_SPLIT] + EVAL_SPLITS}")

        # Create example blocks
        example_blocks: List[str] = []
        for index in reference_indices:
            # Get reference
            reference: Reference = instance.references[index]

            # Construct the passage piece (e.g. "\nPassage: ...\n")
            reference_text: str = (
                self.adapter_spec.reference_prefix + reference.output.text + self.adapter_spec.reference_suffix
            )

            # Construct the question piece (e.g. "\nQuery: ...\n")
            query_text: str = self.adapter_spec.input_prefix + instance.input.text + self.adapter_spec.input_suffix

            # Construct the answer piece (e.g. "\nPrompt: Does the passage above answer the question?\nAnswer: ")
            # If include_output flag is set, answer is appended (e.g. "...\n")
            output_text: str = self.adapter_spec.output_prefix
            if include_output:
                ir_label = self.RANKING_CORRECT_LABEL if CORRECT_TAG in reference.tags else self.RANKING_WRONG_LABEL
                output_text += ir_label + self.adapter_spec.output_suffix
            else:
                output_text = output_text.rstrip()

            # Construct text blocks
            example_block: str = reference_text + query_text + output_text
            example_blocks.append(example_block)

        # Combine the request texts and return
        return self.adapter_spec.instance_prefix.join(example_blocks)
