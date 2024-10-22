import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import replace
from itertools import cycle
from typing import List, Dict, Optional

from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance, TRAIN_SPLIT, EVAL_SPLITS, Reference
from helm.common.general import parallel_map
from helm.common.request import Request
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.benchmark.adaptation.adapters.adapter import Adapter


class InContextLearningAdapter(Adapter, ABC):
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`. It has additional logic surrounding in-context examples.
    """

    @abstractmethod
    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        """
        Given a validation or test `Instance`, generates one or more `RequestState`s.
        """
        pass

    @htrack(None)
    def adapt(self, instances: List[Instance], parallelism: int) -> List[RequestState]:
        """
        Takes a list of `Instance`s and builds a list of corresponding `RequestState`s.
        The reason we don't do this per eval instance is that we create a common set of
        training instances which is shared across all eval instances.
        """
        # Pick out training instances
        all_train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]
        if len(all_train_instances) < self.adapter_spec.max_train_instances:
            hlog(
                f"WARNING: only {len(all_train_instances)} training instances, "
                f"wanted {self.adapter_spec.max_train_instances}"
            )

        # Pick out evaluation instances. This includes both valid and test splits.
        eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]

        hlog(
            f"{len(instances)} instances, "
            f"choosing {self.adapter_spec.max_train_instances}/{len(all_train_instances)} train instances, "
            f"{len(eval_instances)} eval instances"
        )

        # Accumulate all the request states due to adaptation
        all_request_states: List[RequestState] = []
        prompt: Prompt

        for train_trial_index in range(self.adapter_spec.num_train_trials):
            with htrack_block(f"Adapting with train_trial_index={train_trial_index}"):
                all_request_states.extend(
                    self._adapt_trial_index(all_train_instances, train_trial_index, eval_instances, parallelism)
                )

        hlog(f"{len(all_request_states)} requests")
        return all_request_states

    def _adapt_trial_index(
        self,
        all_train_instances: List[Instance],
        train_trial_index: int,
        eval_instances: List[Instance],
        parallelism: int,
    ) -> List[RequestState]:
        training_instances: List[Instance] = self.sample_examples(
            all_train_instances, seed=train_trial_index, sample_train=self.adapter_spec.sample_train
        )
        hlog(f"Sampled {len(training_instances)} examples for trial #{train_trial_index}.")

        def generate_requests_for_training_trial(eval_instance: Instance):
            """Bind some local variables before parallelizing."""
            return self.generate_requests(eval_instance, train_trial_index, training_instances)

        # Generate request_states
        results: List[List[RequestState]] = parallel_map(
            generate_requests_for_training_trial,
            eval_instances,
            parallelism=parallelism,
        )

        # Print out prompts for one instance (useful for debugging)
        if train_trial_index == 0 and len(results) > 0:
            with htrack_block("Sample prompts"):
                for request_state in results[0]:
                    with htrack_block(
                        f"reference index = {request_state.reference_index}, "
                        f"request_mode = {request_state.request_mode}"
                    ):
                        for line in request_state.request.prompt.split("\n"):
                            hlog(line)

        # Flatten and return
        all_request_states: List[RequestState] = [request_state for result in results for request_state in result]
        return self._add_trials(all_request_states)

    def _add_trials(self, request_states: List[RequestState]) -> List[RequestState]:
        """Expand the request states by adding trials."""
        if self.adapter_spec.num_trials <= 1:
            return request_states

        all_request_states: List[RequestState] = request_states.copy()
        for i in range(1, self.adapter_spec.num_trials):
            seed: str = str(i)
            for request_state in request_states:
                request: Request = replace(request_state.request, random=seed)
                all_request_states.append(replace(request_state, request=request))

        assert len(all_request_states) == len(request_states) * self.adapter_spec.num_trials
        return all_request_states

    def sample_examples(
        self, all_train_instances: List[Instance], seed: int, sample_train: bool = True
    ) -> List[Instance]:
        """
        Sample a random set of train instances to use as examples by following the steps below:
        1. Sort the class labels (i.e., correct References) by the number of Instances that belong to the
           class so more common labels are included in the in-context examples. Break ties by shuffling.
        2. Keep sampling one train Instance from each class in the order established in step 1, until
           there are k examples.
        3. If we run out of examples to sample, sample the rest from the Instances that do not have
           class labels.

        Example:

            If we had to sample 2 instances from these train instances:
                Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])]),
                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),
                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),

            The following instances would be selected:

                Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])])
                Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])])

        Returns a new list of randomly sampled train instances.
        """
        # Fix the random seed for reproducibility
        random.seed(seed)
        num_instances_to_sample: int = min(len(all_train_instances), self.adapter_spec.max_train_instances)

        examples: List[Instance] = []
        if not sample_train:
            # Select sequentially from the train set
            examples = all_train_instances[num_instances_to_sample * seed : num_instances_to_sample * (seed + 1)]
            return examples

        unlabeled_instances: List[Instance] = []
        label_to_instances: Dict[str, List[Instance]] = defaultdict(list)
        for instance in all_train_instances:
            if instance.first_correct_reference:
                label_to_instances[instance.first_correct_reference.output.text].append(instance)
            else:
                unlabeled_instances.append(instance)

        # Build Instance counts to labels
        instances: List[Instance]
        counts_to_labels: Dict[int, List[str]] = defaultdict(list)
        for label, instances in sorted(label_to_instances.items()):
            counts_to_labels[len(instances)].append(label)

        sorted_labels: List[str] = []
        # Sort the labels by the number of Instances that belong to them
        for count in sorted(counts_to_labels, reverse=True):
            labels: List[str] = counts_to_labels[count]
            # Break ties by randomly shuffling labels that have the same number of Instances
            random.shuffle(labels)
            sorted_labels.extend(labels)

        labels_iterable = cycle(sorted_labels)
        while num_instances_to_sample > 0:
            next_label: Optional[str] = next(labels_iterable, None)
            if not next_label:
                break

            instances = label_to_instances[next_label]
            # If there are no Instances to sample for this particular label, skip it.
            if len(instances) == 0:
                continue

            # Randomly sample without replacement
            examples.append(instances.pop(random.randrange(len(instances))))
            num_instances_to_sample -= 1

        # If we ran out of Instances with correct References, sample the rest from
        # the pool of Instances without any References
        examples += random.sample(unlabeled_instances, num_instances_to_sample)
        return examples

    def construct_prompt(
        self,
        train_instances: List[Instance],
        eval_instance: Instance,
        include_output: bool,
        reference_index: Optional[int],
    ) -> Prompt:
        """
        Returns a prompt given:
        - the `self.adapter_spec.instructions`
        - the `train_instances` (in-context training examples)
        - the input part of the `eval_instance`
        - the `reference` if `include_output` is true (if reference_index is not None, the reference
        at the given index; otherwise, the first correct reference)

        Fits the prompt within the context window by removing in-context training examples.
        """
        # Instruction text
        instructions_block: str = self.adapter_spec.instructions

        # Text for in-context training instances
        train_instance_blocks: List[str] = [
            self.construct_example_prompt(inst, include_output=True, reference_index=None) for inst in train_instances
        ]

        # Example text
        eval_instance_block: str = self.construct_example_prompt(
            eval_instance, include_output=include_output, reference_index=reference_index
        )

        # Prompt
        prompt = Prompt(
            global_prefix=self.adapter_spec.global_prefix,
            global_suffix=self.adapter_spec.global_suffix,
            instructions_block=instructions_block,
            train_instance_blocks=train_instance_blocks,
            eval_instance_block=eval_instance_block,
            instance_prefix=self.adapter_spec.instance_prefix,
            substitutions=self.adapter_spec.substitutions,
        )

        # Make prompt fit within the context window
        prompt = self._make_prompt_fit(prompt)
        return prompt

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """
        Returns a single example of the prompt. `include_output` controls whether the gold output is included.
        """
        # Input
        result: str = self.adapter_spec.input_prefix + (instance.input.text or "") + self.adapter_spec.input_suffix

        if include_output:
            output: str = self.construct_output(instance, reference_index)
            result += self.adapter_spec.output_prefix + output + self.adapter_spec.output_suffix
        else:
            result += self.adapter_spec.output_prefix.rstrip()

        return result

    def construct_output(self, instance: Instance, reference_index: Optional[int]) -> str:
        """
        Returns the gold output text constructed from correct references.
        If `multi_label` of `AdapterSpec` is true, all correct references are included.
        """
        delimiter: str = ", "
        no_correct_references: str = "n/a"

        output: str
        if reference_index is not None:
            reference = instance.references[reference_index]
            output = reference.output.text
        elif self.adapter_spec.multi_label:
            # Put only the correct references as part as the output
            correct_references: List[Reference] = instance.all_correct_references
            if not correct_references:
                output = no_correct_references
            else:
                output = delimiter.join([correct_reference.output.text for correct_reference in correct_references])
        else:
            first_correct_reference: Optional[Reference] = instance.first_correct_reference
            if not first_correct_reference:
                output = no_correct_references
            else:
                output = first_correct_reference.output.text
        return output

    def _make_prompt_fit(self, prompt: Prompt) -> Prompt:
        """
        The prompt consists of instructions, training instances, and the evaluation input.
        - First, we remove the fewest number of training instances as possible until the prompt fits.
        - Once we hit zero training instances, then we brutally truncate the
          prompt from the right (clearly suboptimal, but hopefully that doesn't
          happen too often).
        Return the prompt that fits.
        """
        # Following what was done for MMLU (https://arxiv.org/abs/2009.03300) to handle prompts that
        # exceed the max context length (https://github.com/hendrycks/test/blob/master/evaluate.py#L58),
        # we remove train instances one by one until it fits within the context window or
        # until we run out of train instances to remove.
        orig_train_instances_count: int = prompt.num_train_instances
        while prompt.num_train_instances > 0:
            if self.window_service.fits_within_context_window(
                text=prompt.text,
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

        # If removing the in-context example is still not enough, we simply truncate the prompt.
        # Following the default truncation strategy used by HuggingFace, we truncate the text from the right.
        text = prompt.text
        truncated_text = self.window_service.truncate_from_right(text, self.adapter_spec.max_tokens)
        if len(truncated_text) < len(text):
            prompt = replace(prompt, truncated_text=truncated_text)
        return prompt
