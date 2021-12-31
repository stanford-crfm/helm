from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

from common.hierarchical_logger import hlog, htrack
from common.request import Request, RequestResult
from .scenario import Instance, Scenario, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take a `Scenario` (a list of `Instance`s) and produce a
    `ScenarioState` (set of `Request`s ).  Instead of having free-form prompt
    hacking, we try to make the process more declarative and systematic.
    Note that an `Instance` could produce many `Request`s (e.g., one for each `Reference`).
    """

    # Prompt starts with instructions
    instructions: str

    # Maximum number of (in-context) training instances to put into the prompt
    max_train_instances: int

    # Maximum number of evaluation instances.  For getting valid numbers, this
    # should be the entire dataset; only reduce this for piloting.
    max_eval_instances: Optional[int]

    # Generate this many outputs (which could be realized by `num_completions`
    # or `top_k_per_token`).
    num_outputs: int

    # Number of trials, where in each trial we choose an independent, random
    # set of training instances.  Used to compute error bars.
    num_train_trials: int

    # Decoding parameters (inherited by `Request`)

    # Model to make the request to
    model: str

    # Temperature to use
    temperature: float

    # When to stop
    stop_sequences: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RequestState:
    """
    A `RequestState` represents a single `Request` made on behalf of an `Instance`.
    It should have all the information that's needed later for a `Metric` to be
    able to understand the `Request` and its `RequestResult`.
    """

    # Which instance we're evaluating
    instance: Instance

    # Which reference of the instance we're evaluating (if any)
    reference_index: Optional[int]

    # Which training set this request is for
    train_trial_index: int

    # The request that is actually made
    request: Request

    # The result of the request (filled in when the request is executed)
    result: Optional[RequestResult]


@dataclass
class ScenarioState:
    """
    A `ScenarioState` represents the output of adaptation.  Contains a set of
    `RequestState` that were created and executed (a `ScenarioState` could be
    pre-execution or post-execution).
    """

    # What strategy we used for adaptation
    adapter_spec: AdapterSpec

    # List of `RequestState`s that were produced by adaptation (and execution)
    request_states: List[RequestState]

    def __post_init__(self):
        # Create derived indices based on `request_states` so it's easier for
        # the `Metric` later to access them.  Two things are produced:
        self.instances = []
        self.request_state_map: Dict[Tuple[int, Instance, Optional[int]], List[RequestState]] = defaultdict(list)

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
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`.  This is where all the prompt hacking for language
    models should go.
    """

    def __init__(self, adapter_spec: AdapterSpec):
        self.adapter_spec = adapter_spec

    @htrack(None)
    def adapt(self, scenario: Scenario) -> ScenarioState:
        """
        Takes a `Scenario` containing a list of instances and builds a list of
        corresponding request_states.  The reason we don't do this per (eval)
        instance is that we create a common set of training instances which is
        shared across all eval instances.
        """
        # Create instances
        instances = scenario.get_instances()

        # Pick out training instances
        all_train_instances = [instance for instance in instances if TRAIN_TAG in instance.tags]
        if len(all_train_instances) < self.adapter_spec.max_train_instances:
            hlog(
                f"WARNING: only {len(all_train_instances)} training instances, "
                f"wanted {self.adapter_spec.max_train_instances}"
            )

        # Pick out evaluation instances.  This includes both valid and test
        # (and any other splits).  We can slice and dice later in defining the
        # metrics.
        eval_instances = [instance for instance in instances if VALID_TAG in instance.tags or TEST_TAG in instance.tags]
        hlog(
            f"{len(instances)} instances, "
            f"choosing {self.adapter_spec.max_train_instances}/{len(all_train_instances)} train instances, "
            f"{len(eval_instances)} eval instances"
        )

        # Accumulate all the request states due to adaptation
        request_states: List[RequestState] = []

        for train_trial_index in range(self.adapter_spec.num_train_trials):
            # Choose a random set of training instances
            random.seed(train_trial_index)
            train_instances = random.sample(
                all_train_instances, min(len(all_train_instances), self.adapter_spec.max_train_instances)
            )

            # Create request_states
            for eval_instance in eval_instances:

                def process(reference_index: Optional[int], reference: Optional[Reference]):
                    prompt = self.construct_prompt(train_instances, eval_instance, reference)
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        temperature=self.adapter_spec.temperature,
                        # TODO: if using single token, then set top_k_per_token instead
                        num_completions=self.adapter_spec.num_outputs,
                        stop_sequences=self.adapter_spec.stop_sequences,
                    )
                    request_state = RequestState(
                        instance=eval_instance,
                        reference_index=reference_index,
                        train_trial_index=train_trial_index,
                        request=request,
                        result=None,
                    )
                    request_states.append(request_state)

                # Request without reference (free-form generation)
                process(None, None)

                # Request for each reference
                for reference_index, reference in enumerate(eval_instance.references):
                    process(reference_index, reference)

        hlog(f"{len(request_states)} requests")
        return ScenarioState(self.adapter_spec, request_states)

    def construct_prompt(
        self, train_instances: List[Instance], eval_instance: Instance, reference: Optional[Reference]
    ) -> str:
        """
        Returns a prompt (string) given:
        - the `self.adapter_spec.instructions`
        - the `train_instances` (in-context training examples)
        - the input part of the `eval_instance`
        - the `reference` (if provided)
        """
        # TODO: support input + output formats
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
            if reference is not None:
                lines.append(output_prefix + reference.output)
            else:
                hlog(f"WARNING: no correct reference for {instance}")
                lines.append(output_prefix + "???")

        # Evaluation instance
        lines.append(input_prefix + eval_instance.input)
        # TODO: removing trailing whitespace
        if reference is None:
            lines.append(output_prefix)
        else:
            lines.append(output_prefix + reference.output)

        return "\n".join(lines)
