from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict
import random

from scenario import Instance, Scenario, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG
from schemas import Request, RequestResult


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take an `Instance` and produce a set of `Request`s (e.g.,
    concatenate instructions and number of training examples) and make one
    request for each reference output).
    """

    instructions: str  # Prompt starts with instructions.
    max_train_instances: int  # Maximum number of (in-context) training instances to put into the prompt
    max_eval_instances: int  # Maximum number of evaluation instances (only reduce this for piloting)
    num_outputs: int  # Generate this many outputs per request

    num_train_trials: int  # Number of random training instances we want to randomize over

    # Decoding parameters
    model: str  # Name of the model we want to query
    temperature: float  # Temperature to use
    stop_sequences: List[str]  # When to stop


@dataclass(frozen=True)
class RequestState:
    """
    A `RequestState` represents a single `Request` made on behalf of an `Instance`.
    It should have all the information that's needed later for a `Metric` to be
    able to understand the `Request` and its `RequestResult`.
    """

    instance: Instance  # Which instance we're evaluating
    reference_index: Optional[int]  # Which reference of the instance we're evaluating (if any)
    train_trial_index: int  # Which training set
    request: Request  # The request that is synthesized
    result: Optional[RequestResult]  # Filled in when we make the call


@dataclass
class ScenarioState:
    """All the `RequestState` results that come about from evaluating a particular scenario."""

    adapter_spec: AdapterSpec
    request_states: List[RequestState]

    def __post_init__(self):
        # Create an index for `instances` and `request_states`.
        self.instances = []
        self.request_state_map: Dict[Tuple[int, Instance, reference_index], List[RequestState]] = defaultdict(list)
        instances_set = set()
        for request_state in self.request_states:
            instances_set.add(request_state.instance)
            key = (request_state.train_trial_index, request_state.instance, request_state.reference_index)
            self.request_state_map[key].append(request_state)
        self.instances = list(instances_set)

    def get_request_states(
        self, train_trial_index: int, instance: Instance, reference_index: Optional[int]
    ) -> List[RequestState]:
        return self.request_state_map.get((train_trial_index, instance, reference_index), [])


class Adapter:
    """An `Adapter`"""

    def __init__(self, adapter_spec: AdapterSpec):
        self.adapter_spec = adapter_spec

    def adapt(self, scenario: Scenario) -> ScenarioState:
        """
        Takes a `Scenario` containing a list of instances and builds a list of
        corresponding request_states.  The reason we don't do this per (eval)
        instance is that we create a common set of training instances which is
        shared across all eval instances.
        """
        # Create instances
        instances = scenario.get_instances()

        # Choose training instances and evaluation instances
        all_train_instances = [instance for instance in instances if TRAIN_TAG in instance.tags]
        if len(all_train_instances) < self.adapter_spec.max_train_instances:
            print(
                f"WARNING: only {len(all_train_instances)} training instances, wanted {self.adapter_spec.max_train_instances}"
            )
        eval_instances = [instance for instance in instances if VALID_TAG in instance.tags or TEST_TAG in instance.tags]
        print(
            f"{len(instances)} instances, choosing {self.adapter_spec.max_train_instances}/{len(all_train_instances)} train instances, {len(eval_instances)} eval instances"
        )

        request_states: List[RequestState] = []

        for train_trial_index in range(self.adapter_spec.num_train_trials):
            # Choose a random set of training instances
            random.seed(train_trial_index)
            train_instances = random.sample(
                all_train_instances, min(len(all_train_instances), self.adapter_spec.max_train_instances)
            )

            # Create request_states
            for eval_instance in eval_instances:
                for reference_index, reference in [(None, None)] + list(enumerate(eval_instance.references)):
                    prompt = self.construct_prompt(train_instances, eval_instance, reference)
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        temperature=self.adapter_spec.temperature,
                        topK=self.adapter_spec.num_outputs,
                        stopSequences=self.adapter_spec.stop_sequences,
                    )
                    request_state = RequestState(
                        instance=eval_instance,
                        reference_index=reference_index,
                        train_trial_index=train_trial_index,
                        request=request,
                        result=None,
                    )
                    request_states.append(request_state)

        print(f"{len(request_states)} requests")
        return ScenarioState(self.adapter_spec, request_states)

    def construct_prompt(
        self, train_instances: List[Instance], eval_instance: Instance, reference: Optional[Reference]
    ) -> str:
        """
        Returns a prompt (string) given `self.adapter_spec.instructions`,
        `train_instances` (in-context training examples), the input part of the
        `eval_instance`, and optionally the `reference`.
        """
        # TODO: make this configurable if desired
        # TODO: what happens if we have multiline text?
        input_prefix = "Input: "
        output_prefix = "Output: "

        # Instructions
        lines = [self.adapter_spec.instructions]

        # In-context training instances
        for instance in train_instances:
            lines.append("")
            lines.append(input_prefix + instance.input)
            # Put only the correct reference as the output
            reference = instance.first_correct_reference
            lines.append(output_prefix + reference.output)

        # Evaluation instance
        lines.append(input_prefix + eval_instance.input)
        # TODO: removing trailing whitespace
        lines.append(output_prefix)

        return "\n".join(lines)
