import random
from dataclasses import dataclass, field, replace
from itertools import cycle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, OrderedDict

import numpy as np

from common.general import serialize, indent_lines, format_text_lines, parallel_map, flatten_list
from common.hierarchical_logger import hlog, htrack, htrack_block
from common.request import Request, RequestResult
from common.tokenization_request import TokenizationToken
from .scenarios.scenario import Instance, TRAIN_SPLIT, EVAL_SPLITS
from .window_services.window_service import WindowService, EncodeResult
from .window_services.window_service_factory import WindowServiceFactory
from .window_services.tokenizer_service import TokenizerService

# Methods of adaptation
ADAPT_LANGUAGE_MODELING = "language_modeling"
ADAPT_MULTIPLE_CHOICE_JOINT = "multiple_choice_joint"
ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL = "multiple_choice_separate_original"
ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED = "multiple_choice_separate_calibrated"
ADAPT_GENERATION = "generation"
ADAPT_RANKING_BINARY = "ranking_binary"


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

    # Prompt starts with instructions
    instructions: str = ""

    # What goes before the input
    input_prefix: str = "Input: "

    # What goes before the input (for multiple choice)
    reference_prefix: str = "\nA. "

    # What goes before the output
    output_prefix: str = "\nOutput: "

    # What goes before each Instance in the constructed prompt
    instance_prefix: str = "\n\n"

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

    # Random string (used concretely to bypass cache / see diverse results)
    random: Optional[str] = None


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

    # Which request mode ("original" or "calibration") of the instance we're evaluating (if any)
    # (for ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED)
    request_mode: Optional[str]

    # Which training set this request is for
    train_trial_index: int

    # How to map the completion text back to a real output (e.g., for multiple choice, "B" => "the second choice")
    output_mapping: Optional[Dict[str, str]]

    # The request that is actually made
    request: Request

    # The result of the request (filled in when the request is executed)
    result: Optional[RequestResult]

    # The number of in-context examples used
    num_in_context_examples: int

    # Whether the input of the corresponding Instance was truncated to fit the model's context window for this Request
    # (input_truncated == True implies num_in_context_examples == 0)
    input_truncated: bool

    # The number of initial tokens that will be ignored when computing language modeling metrics
    num_conditioning_tokens: int = 0

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
        self.request_state_map: Dict[Tuple[int, Instance, Optional[int]], List[RequestState]] = defaultdict(list)

        # Python doesn't support an ordered set, so use an OrderedDict instead to maintain insertion order
        instances_set: Dict[Instance, None] = OrderedDict()
        for request_state in self.request_states:
            instances_set[request_state.instance] = None
            key = (request_state.train_trial_index, request_state.instance, request_state.reference_index)
            self.request_state_map[key].append(request_state)
        self.instances: List[Instance] = list(instances_set.keys())

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


@dataclass
class Prompt:
    """Result of prompt construction."""

    # The prompt
    text: str

    # Number of in-context examples in the prompt
    num_in_context_examples: int

    # Whether the input of the corresponding Instance was truncated to fit the model's context window for this Request
    # (input_truncated == True implies num_in_context_examples == 0)
    input_truncated: bool


@dataclass(frozen=True)
class Processor:
    """Constructs a prompt for each example."""

    adapter_spec: AdapterSpec
    window_service: WindowService

    train_instances: List[Instance]
    train_trial_index: int

    def process(self, eval_instance: Instance) -> List[RequestState]:
        # Get adaptation method.
        method = self.adapter_spec.method

        # Generate request states according to the chosen method.
        if method == ADAPT_GENERATION:
            return self.adapt_generation(eval_instance)
        elif method == ADAPT_MULTIPLE_CHOICE_JOINT:
            return self.adapt_multiple_choice_joint(eval_instance)
        elif method in [ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED]:
            return self.adapt_multiple_choice_separate(eval_instance)
        elif method == ADAPT_RANKING_BINARY:
            return self.adapt_ranking_binary(eval_instance)
        raise ValueError(f"Invalid method: {method}")

    def adapt_generation(self, eval_instance: Instance) -> List[RequestState]:
        prompt = self.construct_prompt(self.train_instances, eval_instance, include_output=False, reference_index=None)
        request = Request(
            model=self.adapter_spec.model,
            prompt=prompt.text,
            num_completions=self.adapter_spec.num_outputs,
            temperature=self.adapter_spec.temperature,
            max_tokens=self.adapter_spec.max_tokens,
            stop_sequences=self.adapter_spec.stop_sequences,
            random=self.adapter_spec.random,
        )
        request_state = RequestState(
            instance=eval_instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=self.train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_in_context_examples=prompt.num_in_context_examples,
            input_truncated=prompt.input_truncated,
        )
        return [request_state]

    def adapt_multiple_choice_joint(self, eval_instance: Instance) -> List[RequestState]:
        prompt = self.construct_prompt(self.train_instances, eval_instance, include_output=False, reference_index=None)
        output_mapping = dict(
            (self.get_reference_prefix("A", reference_index), reference.output)
            for reference_index, reference in enumerate(eval_instance.references)
        )
        request = Request(
            model=self.adapter_spec.model,
            prompt=prompt.text,
            num_completions=1,
            top_k_per_token=self.adapter_spec.num_outputs,
            temperature=0,
            max_tokens=1,
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
            num_in_context_examples=prompt.num_in_context_examples,
            input_truncated=prompt.input_truncated,
        )
        return [request_state]

    def adapt_multiple_choice_separate(self, eval_instance: Instance) -> List[RequestState]:
        request_states = []
        for reference_index, reference in enumerate(eval_instance.references):
            # Explanation for request_modes:
            # - ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL: each answer choice sentence is
            # scored independently, where the score is the sentence probability
            # normalized by sentence length.
            # - ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED: each answer choice sentence is
            # scored independently, where the score is the sentence probability
            # normalized by the unconditional sentence probability.
            # Details refer to Section 2.4 of GPT-3 paper (https://arxiv.org/pdf/2005.14165.pdf))
            if self.adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL:
                request_modes = ["original"]
            elif self.adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED:
                request_modes = ["original", "calibration"]
            else:
                raise ValueError(f"Unknown adapter method: {self.adapter_spec.method}")

            for request_mode in request_modes:
                if request_mode == "original":
                    prompt = self.construct_prompt(
                        self.train_instances, eval_instance, include_output=True, reference_index=reference_index,
                    )
                elif request_mode == "calibration":
                    # For calibration purpose, we compute the logprobs of the reference
                    # without train instances and the input question.
                    eval_instance_calibration = replace(eval_instance, input="Answer:")
                    prompt = self.construct_prompt(
                        [], eval_instance_calibration, include_output=True, reference_index=reference_index,
                    )
                else:
                    raise ValueError(f"Unknown request mode: {request_mode}")
                request = Request(
                    model=self.adapter_spec.model,
                    prompt=prompt.text,
                    num_completions=1,
                    temperature=0,
                    max_tokens=0,
                    stop_sequences=[],
                    echo_prompt=True,
                )
                request_state = RequestState(
                    instance=eval_instance,
                    reference_index=reference_index,
                    request_mode=request_mode,
                    train_trial_index=self.train_trial_index,
                    output_mapping=None,
                    request=request,
                    result=None,
                    num_in_context_examples=prompt.num_in_context_examples,
                    input_truncated=prompt.input_truncated,
                )
                request_states.append(request_state)
        return request_states

    def adapt_ranking_binary(self, eval_instance: Instance) -> List[RequestState]:
        """ TODO: Not ready. """
        request_states = []
        for reference_index, reference in enumerate(eval_instance.references):
            prompt = self.construct_prompt(
                self.train_instances, eval_instance, include_output=True, reference_index=reference_index
            )  # TODO: This needs to be replaced.
            request = Request(
                model=self.adapter_spec.model,
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
                request_mode=None,
                train_trial_index=self.train_trial_index,
                output_mapping=None,
                request=request,
                result=None,
                num_in_context_examples=prompt.num_in_context_examples,
                input_truncated=prompt.input_truncated,
            )
            request_states.append(request_state)
        return request_states

    def construct_prompt(
        self,
        train_instances: List[Instance],
        eval_instance: Instance,
        include_output: bool,
        reference_index: Optional[int],
    ) -> Prompt:
        """
        Returns a prompt (string) given:
        - the `self.adapter_spec.instructions`
        - the `train_instances` (in-context training examples)
        - the input part of the `eval_instance`
        - the `reference` if `include_output` is true (if reference_index is not None, the reference
        at the given index; otherwise, the first correct reference)

        Fits the prompt within the context window by removing in-context training examples.
        """
        assert include_output or reference_index is None

        def construct_prompt_helper(train_instances: List[Instance]) -> str:
            # Instructions
            blocks = []

            if self.adapter_spec.instructions:
                blocks.append(self.adapter_spec.instructions)

            # In-context training instances
            for instance in train_instances:
                blocks.append(self.construct_example_prompt(instance, include_output=True, reference_index=None))

            blocks.append(
                self.construct_example_prompt(
                    eval_instance, include_output=include_output, reference_index=reference_index
                )
            )

            return self.adapter_spec.instance_prefix.join(blocks)

        orig_train_instances_count: int = len(train_instances)
        prompt: str = construct_prompt_helper(train_instances)

        # Following what was done for MMLU (https://arxiv.org/abs/2009.03300) to handle prompts that
        # exceed the max context length (https://github.com/hendrycks/test/blob/master/evaluate.py#L58),
        # we remove train instances one by one until it fits within the context window or
        # until we run out of train instances to remove.
        while len(train_instances) > 0:
            if self.window_service.fits_within_context_window(
                text=prompt, expected_completion_token_length=self.adapter_spec.max_tokens,
            ):
                return Prompt(prompt, num_in_context_examples=len(train_instances), input_truncated=False)

            train_instances = train_instances[:-1]
            prompt = construct_prompt_helper(train_instances)  # TODO: Needs re-factor.

        removed_train_instances_count: int = orig_train_instances_count - len(train_instances)
        if removed_train_instances_count > 0:
            hlog(
                f"The original constructed prompt exceeded the max context length. Removed "
                f"{removed_train_instances_count} in-context examples to fit it within the context window."
            )

        # If removing the in-context example is still not enough, we simply truncate the prompt.
        # Following the default truncation strategy used by HuggingFace, we truncate the text from the right.
        original_length = len(prompt)
        prompt = self.window_service.truncate_from_right(prompt, self.adapter_spec.max_tokens)
        input_truncated = len(prompt) < original_length
        return Prompt(prompt, num_in_context_examples=len(train_instances), input_truncated=input_truncated)

    def construct_example_prompt(self, instance: Instance, include_output: bool, reference_index: Optional[int]) -> str:
        """Return a list of lines corresponding to this example (part of the prompt)."""

        # Input
        result = self.adapter_spec.input_prefix + instance.input

        # References (optionally) and output
        if self.adapter_spec.method == ADAPT_MULTIPLE_CHOICE_JOINT:
            # If multiple choice, include the references
            output = "n/a"
            for reference_index, reference in enumerate(instance.references):
                prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)
                result += prefix + reference.output
                if reference.is_correct and output == "n/a":
                    output = self.get_reference_prefix("A", reference_index)
        else:
            if reference_index is None:
                # Put only the correct reference as the output
                correct_reference = instance.first_correct_reference
                output = correct_reference.output if correct_reference is not None else "n/a"
            else:
                reference = instance.references[reference_index]
                output = reference.output

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

    def __init__(self, adapter_spec: AdapterSpec, tokenizer_service: TokenizerService):
        self.adapter_spec: AdapterSpec = adapter_spec
        self.window_service: WindowService = WindowServiceFactory.get_window_service(
            adapter_spec.model, tokenizer_service
        )

    def sample_instances(self, instances: List[Instance]) -> List[Instance]:
        """
        Leave the train instances alone.
        For the eval instances, keep at most `max_eval_instances`.
        Return the resulting train and eval instances.
        """
        all_train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]

        all_eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        if (
            self.adapter_spec.max_eval_instances is not None
            and len(all_eval_instances) > self.adapter_spec.max_eval_instances
        ):
            # Pick the first `self.adapter_spec.max_eval_instances`.
            # The random sampling includes instances monotonically.
            np.random.seed(0)
            selected_eval_instances = list(
                np.random.choice(all_eval_instances, self.adapter_spec.max_eval_instances, replace=False,)
            )  # type: ignore
        else:
            selected_eval_instances = all_eval_instances

        hlog(
            f"{len(instances)} instances, "
            f"{len(all_train_instances)} train instances, "
            f"{len(selected_eval_instances)}/{len(all_eval_instances)} eval instances"
        )

        return all_train_instances + selected_eval_instances

    @htrack(None)
    def adapt(self, instances: List[Instance], parallelism: int) -> ScenarioState:
        """
        Takes a a list of `Instance`s and builds a list of corresponding `RequestState`s.
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

        if self.adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            # Use the LM-specific method to adapt LM scenarios
            all_request_states = self.adapt_language_modeling(eval_instances, parallelism)
        else:
            for train_trial_index in range(self.adapter_spec.num_train_trials):
                with htrack_block(f"Adapting with train_trial_index={train_trial_index}"):
                    all_request_states.extend(
                        self.adapt_trial_index(all_train_instances, train_trial_index, eval_instances, parallelism)
                    )

        hlog(f"{len(all_request_states)} requests")
        return ScenarioState(self.adapter_spec, all_request_states)

    def adapt_trial_index(
        self,
        all_train_instances: List[Instance],
        train_trial_index: int,
        eval_instances: List[Instance],
        parallelism: int,
    ) -> List[RequestState]:
        train_instances: List[Instance] = self.sample_examples(all_train_instances, seed=train_trial_index)

        # Create request_states
        processor = Processor(
            adapter_spec=self.adapter_spec,
            window_service=self.window_service,
            train_instances=train_instances,
            train_trial_index=train_trial_index,
        )
        results: List[List[RequestState]] = parallel_map(
            processor.process, eval_instances, parallelism=parallelism,
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
        all_request_states: List[RequestState] = []
        for result_index, result in enumerate(results):
            all_request_states.extend(result)
        return [request_state for result in results for request_state in result]

    def sample_examples(self, all_train_instances: List[Instance], seed: int) -> List[Instance]:
        """
        Sample a random set of train instances to use as examples by following the steps below:
        1. Sort the class labels (correct References) by the number of Instances that belong to the class.
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

        unlabeled_instances: List[Instance] = []
        label_to_instances: Dict[str, List[Instance]] = defaultdict(list)

        for instance in all_train_instances:
            if instance.first_correct_reference:
                label_to_instances[instance.first_correct_reference.output].append(instance)
            else:
                unlabeled_instances.append(instance)

        # Sort the labels by the number of Instances that belong to them
        sorted_labels: List[str] = [
            key for key, _ in sorted(label_to_instances.items(), key=lambda x: len(x[1]), reverse=True)
        ]
        labels_iterable = cycle(sorted_labels)

        examples: List[Instance] = []
        while num_instances_to_sample > 0:
            next_label: Optional[str] = next(labels_iterable, None)
            if not next_label:
                break

            instances: List[Instance] = label_to_instances[next_label]
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

    def fits_tokens_within_context_window(
        self,
        conditioning_tokens: List[TokenizationToken],
        pred_tokens: List[TokenizationToken],
        max_req_len: int,
        text: Optional[str] = None,
    ) -> Tuple[str, List[TokenizationToken]]:
        """
        This method is used for adapting instances for language modeling scenarios.
        For some tokenizers (e.g. AI21), decoding then encoding k tokens may result
        in > k tokens. This method trims the tokens and check with the tokenizer
        repeatedly until they fit in the context window.

        For models using the GPT-2 tokenizer, conditioning_tokens and pred_tokens
        are integers; for AI21 models, the tokens are TokenizationTokens.
        """
        prompt: str = self.window_service.decode(conditioning_tokens + pred_tokens, text)
        prompt_length: int = len(self.window_service.encode(prompt).tokens)

        # If the prompt is too long, removes the overflowing tokens.
        # Since encoding might generate extra tokens, we need to repeat this until prompt_length <= max_req_len.
        # For AI21, for example, this happens especially frequently when a document contains different types of
        # whitespace characters because some whitespaces are tokenized to multiple tokens and the others
        # are tokenized to a single token. However, the AI21 tokenizer seems to normalize all types
        # of whitespaces to the same whitespace character.
        #
        # e.g. original text: ",  (", which is tokenized to:
        # [('▁', 0, 0), (',', 0, 1), ('▁▁', 1, 3), ('(', 3, 4)]
        # normalized text: ",  (", which is tokenized to:
        # [('▁', 0, 0), (',', 0, 1), ('▁', 1, 2), ('▁', 2, 3), ('(', 3, 4)]
        while prompt_length > max_req_len:
            # Trims the extra (prompt_length - max_req_len) tokens
            pred_tokens = pred_tokens[: -(prompt_length - max_req_len)]
            prompt = self.window_service.decode(conditioning_tokens + pred_tokens, text)
            prompt_length = len(self.window_service.encode(prompt).tokens)

        return prompt, pred_tokens

    def construct_language_modeling_prompt(
        self,
        conditioning_tokens: List[TokenizationToken],
        pred_tokens: List[TokenizationToken],
        max_req_len: int,
        text: str,
    ) -> Tuple[str, int]:
        """
        Some subwords/symbols might translate to multiple tokens. e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"].

        When a subword of this type happens to be the last token of a chunk, we need to strip the leading and
        trailing bytes to ensure the prompt is a valid string.

        Since some tokens are removed, we also need to recompute num_conditioning_tokens.

        For models using the GPT-2 tokenizer, conditioning_tokens and pred_tokens are integers; for AI21
        models, the tokens are TokenizationTokens.

        text is the normalized text fed to decode(). Some tokenizers (e.g. AI21) need this field for decoding.
        """
        raw_prompt: str
        raw_prompt, pred_tokens = self.fits_tokens_within_context_window(
            conditioning_tokens, pred_tokens, max_req_len, text
        )

        prompt: str = raw_prompt.strip("\ufffd")
        # If there are no byte tokens, avoid API calls
        if len(prompt) == len(raw_prompt):
            num_conditioning_tokens = len(conditioning_tokens)
        else:
            num_leading_byte_tokens: int = max_req_len - len(
                self.window_service.encode(raw_prompt.lstrip("\ufffd")).tokens
            )
            num_trailing_byte_tokens: int = max_req_len - len(
                self.window_service.encode(raw_prompt.rstrip("\ufffd")).tokens
            )

            # There are no string tokens to predict
            if num_trailing_byte_tokens >= len(pred_tokens):
                num_conditioning_tokens = len(self.window_service.encode(prompt).tokens)
            # There are no conditioning string tokens
            elif num_leading_byte_tokens >= len(conditioning_tokens):
                num_conditioning_tokens = 1
            else:
                num_conditioning_tokens = len(conditioning_tokens) - num_leading_byte_tokens
        return prompt, num_conditioning_tokens

    def adapt_language_modeling(self, instances: List[Instance], parallelism: int) -> List[RequestState]:
        """ Code is adapted from:

        https://github.com/EleutherAI/lm_perplexity/blob/main/lm_perplexity/utils.py
        """
        max_sequence_length: int = self.window_service.max_sequence_length
        max_request_length: int = self.window_service.max_request_length
        prefix_token: str = self.window_service.prefix_token

        # TODO: does not support multiprocessing
        def process(instance: Instance) -> List[RequestState]:
            encode_result: EncodeResult = self.window_service.encode(instance.input)
            tokens: List[TokenizationToken] = encode_result.tokens
            text: str = encode_result.text

            request_states: List[RequestState] = []
            num_predicted_tokens: int = 0

            # Special handling for first window: predict all tokens
            # Example for GPT-3:
            # Raw token sequence format: [<str_tok1>, <str_tok2>, ..., <byte_tok1>, ...]
            # (total length <= max_sequence_length)
            # Convert it to: [<eot>, <str_tok1>, <str_tok2>, ...]
            # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
            # Num_conditioning_tokens = 1
            # Example: ["Hello", " world", "bytes:\xe2\x80"] => "<eot>Hello world"
            #
            # Note: There are trailing byte tokens in the raw sequence because some subwords/symbols might translate to
            # multiple tokens (e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"]) and we chunk documents by token, not by word.

            # Uses `max_sequence_length` instead of `max_request_length` here because `prefix_token` will be prepended
            # to the sequence later. This is the only place where `max_sequence_length` is used.
            first_seq_len: int = min(max_sequence_length, len(tokens))
            prompt, num_conditioning_tokens = self.construct_language_modeling_prompt(
                self.window_service.encode(prefix_token).tokens, tokens[:first_seq_len], max_request_length, text
            )
            request = Request(
                model=self.adapter_spec.model,
                prompt=prompt,
                num_completions=1,
                temperature=0,
                max_tokens=self.adapter_spec.max_tokens,  # usually this is zero
                stop_sequences=self.adapter_spec.stop_sequences,
                echo_prompt=True,
                random=self.adapter_spec.random,
            )
            request_state = RequestState(
                instance=instance,
                reference_index=None,
                request_mode=None,
                train_trial_index=0,
                output_mapping=None,
                request=request,
                result=None,
                num_conditioning_tokens=1 if len(prefix_token) > 0 else 0,
                num_in_context_examples=self.adapter_spec.max_train_instances,
                input_truncated=False,
            )
            request_states.append(request_state)
            num_predicted_tokens += first_seq_len

            while num_predicted_tokens < len(tokens):
                # Example for GPT-3:
                # Raw token sequence format:
                # [<cond_byte1>, ..., <cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, ..., <pred_byte1>, ...]
                # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
                #
                # Convert it to: [<cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, <pred_str_tok2>. ...]
                # (total length <= max_req_len = max_sequence_length+1 for GPT-3)
                #
                # Example: conditioning_tokens=["bytes:\x99", "Exc"], pred_tokens=["use", " me", "bytes:\xe2\x80"] =>
                # prompt="Excuse me", num_conditioning_tokens = 1

                # The upper bound is `max_req_len - 1` because there will be at least 1 conditioning tokens.
                window_pred_len: int = min(len(tokens) - num_predicted_tokens, max_request_length - 1)
                window_end: int = num_predicted_tokens + window_pred_len
                conditioning_tokens: List[TokenizationToken] = tokens[
                    window_end - max_request_length : num_predicted_tokens
                ]
                pred_tokens: List[TokenizationToken] = tokens[num_predicted_tokens:window_end]
                prompt, num_conditioning_tokens = self.construct_language_modeling_prompt(
                    conditioning_tokens, pred_tokens, max_request_length, text
                )

                request = Request(
                    model=self.adapter_spec.model,
                    prompt=prompt,
                    num_completions=1,
                    temperature=0,
                    max_tokens=self.adapter_spec.max_tokens,  # usually this is zero
                    stop_sequences=self.adapter_spec.stop_sequences,
                    echo_prompt=True,
                )
                request_state = RequestState(
                    instance=instance,
                    reference_index=None,
                    request_mode=None,
                    train_trial_index=0,
                    output_mapping=None,
                    request=request,
                    result=None,
                    num_conditioning_tokens=num_conditioning_tokens,
                    num_in_context_examples=self.adapter_spec.max_train_instances,
                    input_truncated=False,
                )
                request_states.append(request_state)
                num_predicted_tokens += window_pred_len

            return request_states

        results: List[List[RequestState]] = parallel_map(process, instances, parallelism)
        return flatten_list(results)
