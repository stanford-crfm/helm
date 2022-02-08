import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from common.general import serialize, indent_lines, format_text_lines
from common.hierarchical_logger import hlog, htrack, htrack_block
from common.request import Request, RequestResult
from .scenario import Instance, Scenario, TRAIN_TAG, VALID_TAG, TEST_TAG


# Methods of adaptation
ADAPT_LANGUAGE_MODELING = "language_modeling"
ADAPT_MULTIPLE_CHOICE = "multiple_choice"
ADAPT_GENERATION = "generation"


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take a `Scenario` (a list of `Instance`s) and produce a
    `ScenarioState` (set of `Request`s ).  Instead of having free-form prompt
    hacking, we try to make the process more declarative and systematic.
    Note that an `Instance` could produce many `Request`s (e.g., one for each `Reference`).
    """

    # Method of adaptation
    method: str

    # Conditioning prefix whose logprob is ignored
    conditioning_prefix: str = ""

    # Prompt starts with instructions
    instructions: str = ""

    # What goes before the input
    input_prefix: str = "Input: "

    # What goes before the input (for multiple choice)
    reference_prefix: str = "\nA. "

    # What goes before the output
    output_prefix: str = "\nOutput: "

    # Maximum number of (in-context) training instances to put into the prompt
    max_train_instances: int = 5

    # Maximum number of evaluation instances.  For getting valid numbers, this
    # should be the entire dataset; only reduce this for piloting.
    max_eval_instances: Optional[int] = None

    # Generate this many outputs (which could be realized by `num_completions`
    # or `top_k_per_token`).
    num_outputs: int = 5

    # Number of trials, where in each trial we choose an independent, random
    # set of training instances.  Used to compute error bars.
    num_train_trials: int = 1

    # Decoding parameters (inherited by `Request`)

    # Model to make the request to
    model: str = "openai/davinci"

    # Temperature to use
    temperature: float = 1

    # Maximum number of tokens to generate
    max_tokens: int = 100

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

    # How to map the completion text back to a real output (e.g., for multiple choice, "B" => "the second choice")
    output_mapping: Optional[Dict[str, str]]

    # The request that is actually made
    request: Request

    # The result of the request (filled in when the request is executed)
    result: Optional[RequestResult]

    def render_lines(self) -> List[str]:
        output = [f"train_trial_index: {self.train_trial_index}"]
        if self.reference_index:
            output.append(f"reference_index: {self.reference_index}")

        output.append("instance {")
        output.extend(indent_lines(self.instance.render_lines()))
        output.append("}")

        # Part of request but render multiline
        output.append("request.prompt {")
        output.extend(indent_lines(format_text_lines(self.request.prompt)))
        output.append("}")

        output.append("request {")
        output.extend(indent_lines(serialize(self.request)))
        output.append("}")

        if self.result:
            output.append("result {")
            output.extend(indent_lines(self.result.render_lines()))
            output.append("}")

        return output


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

    def render_lines(self) -> List[str]:
        total: int = len(self.request_states)
        result = ["adapter_spec {"]
        result.extend(indent_lines(serialize(self.adapter_spec)))
        result.append("}")

        for i, request_state in enumerate(self.request_states):
            result.append(f"request_state {i} ({total} total) {{")
            result.extend(indent_lines(request_state.render_lines()))
            result.append("}")

        return result


class Adapter:
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`.  This is where all the prompt hacking for language
    models should go.

    Recall that each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

    There are several ways to adapt:

    1. [language_modeling] We don't use the references (even if they exist), just feed the input:

        <input>

    2. [multiple_choice] We can define a label (e.g., letter) for each reference:

        <instructions>

        <input>                  # train
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: C

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:

        In general, each example is:

            <input_prefix><input><reference_prefixes[0]><reference><output_prefix><output>

    3. [generation] Just let the language model try to generate the output.

        <instructions>

        Input: <input>                  # train
        Output: <reference>

        Input: <input>                  # test
        Output:

        In general, each example is:

            <input_prefix><input><output_prefix><output>

    Later, we can create multiple requests, one for each reference and try to
    score their probabilities to solve multiple choice problems (but have to
    deal with references being of different lengths).
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
        with htrack_block("scenario.get_instances"):
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
        if self.adapter_spec.max_eval_instances is not None:
            eval_instances = eval_instances[: self.adapter_spec.max_eval_instances]
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
            for eval_index, eval_instance in enumerate(eval_instances):
                prompt = self.construct_prompt(train_instances, eval_instance)

                # Just print one prompt (useful for debugging)
                if train_trial_index == 0 and eval_index == 0:
                    with htrack_block("Sample prompt"):
                        for line in prompt.split("\n"):
                            hlog(line)

                # Define the request
                method = self.adapter_spec.method
                if method == ADAPT_LANGUAGE_MODELING:
                    output_mapping = None
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        num_completions=1,
                        temperature=0,
                        max_tokens=self.adapter_spec.max_tokens,
                        stop_sequences=self.adapter_spec.stop_sequences,
                        echo_prompt=True,
                    )
                elif method == ADAPT_MULTIPLE_CHOICE:
                    output_mapping = dict(
                        (self.get_reference_prefix("A", reference_index), reference.output)
                        for reference_index, reference in enumerate(eval_instance.references)
                    )
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        num_completions=1,
                        top_k_per_token=self.adapter_spec.num_outputs,
                        temperature=0,
                        max_tokens=1,
                        stop_sequences=[],
                    )
                elif method == ADAPT_GENERATION:
                    output_mapping = None
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        num_completions=self.adapter_spec.num_outputs,
                        temperature=self.adapter_spec.temperature,
                        max_tokens=self.adapter_spec.max_tokens,
                        stop_sequences=self.adapter_spec.stop_sequences,
                    )
                else:
                    raise ValueError(f"Invalid method: {method}")

                request_state = RequestState(
                    instance=eval_instance,
                    reference_index=None,
                    train_trial_index=train_trial_index,
                    output_mapping=output_mapping,
                    request=request,
                    result=None,
                )
                request_states.append(request_state)

        hlog(f"{len(request_states)} requests")
        return ScenarioState(self.adapter_spec, request_states)

    def construct_prompt(self, train_instances: List[Instance], eval_instance: Instance) -> str:
        """
        Returns a prompt (string) given:
        - the `self.adapter_spec.instructions`
        - the `train_instances` (in-context training examples)
        - the input part of the `eval_instance`
        - the `reference` (if provided)
        """
        # Instructions
        blocks = []

        if self.adapter_spec.instructions:
            blocks.append(self.adapter_spec.instructions)

        # In-context training instances
        for instance in train_instances:
            blocks.append(self.construct_example_prompt(instance, include_output=True))

        blocks.append(self.construct_example_prompt(eval_instance, include_output=False))

        return self.adapter_spec.conditioning_prefix + "\n\n".join(blocks)

    def construct_example_prompt(self, instance: Instance, include_output: bool) -> str:
        """Return a list of lines corresponding to this example (part of the prompt)."""

        # Input
        result = self.adapter_spec.input_prefix + instance.input

        # References (optionally) and output
        if self.adapter_spec.method == ADAPT_MULTIPLE_CHOICE:
            # If multiple choice, include the references
            output = "n/a"
            for reference_index, reference in enumerate(instance.references):
                prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
                result += prefix + reference.output
                if reference.is_correct and output == "n/a":
                    output = self.get_reference_prefix("A", reference_index)
        else:
            # Put only the correct reference as the output
            correct_reference = instance.first_correct_reference
            output = correct_reference.output if correct_reference is not None else "n/a"

        if include_output:
            result += self.adapter_spec.output_prefix + output
        else:
            result += self.adapter_spec.output_prefix.rstrip()

        return result

    def get_reference_prefix(self, prefix: str, i: int) -> str:
        """
        Example: prefix = "\nA. ", i = 2, return "\nC. "
        """
        return prefix.replace("A", chr(ord("A") + i))
