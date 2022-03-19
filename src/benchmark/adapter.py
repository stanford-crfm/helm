import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, OrderedDict

from common.general import serialize, indent_lines, format_text_lines
from common.hierarchical_logger import hlog, htrack, htrack_block
from common.request import Request, RequestResult
from proxy.tokenizer.tokenizer import Tokenizer
from proxy.tokenizer.tokenizer_factory import TokenizerFactory
from .tokenizer_service import TokenizerService
from .scenario import Instance, TRAIN_SPLIT, EVAL_SPLITS

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

    # The number of initial tokens that will be ignored when computing langauge modeling metrics
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

        instances_set = set()
        for request_state in self.request_states:
            instances_set.add(request_state.instance)
            key = (request_state.train_trial_index, request_state.instance, request_state.reference_index)
            self.request_state_map[key].append(request_state)
        self.instances: List[Instance] = list(instances_set)

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

    def __init__(self, adapter_spec: AdapterSpec, tokenizer_service: TokenizerService):
        self.adapter_spec: AdapterSpec = adapter_spec
        self.tokenizer: Tokenizer = TokenizerFactory.get_tokenizer(adapter_spec.model, tokenizer_service)

    @htrack(None)
    def adapt(self, instances: List[Instance]) -> ScenarioState:
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
        # We can slice and dice later in defining the metrics.
        eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        if self.adapter_spec.max_eval_instances is not None:
            # Build a dict of instance IDs to instances before we pick self.adapter_spec.max_eval_instances
            # number of instances, so we can include all the perturbed versions of the instances
            # we choose in the eval set.
            id_to_instances: OrderedDict[Optional[str], List[Instance]] = OrderedDict()
            for instance in eval_instances:
                if instance.id in id_to_instances:
                    id_to_instances[instance.id].append(instance)
                else:
                    id_to_instances[instance.id] = [instance]

            # Pick the first `self.adapter_spec.max_eval_instances` instance IDs and
            # include all their instances in the final set of eval instances.
            eval_instances = []
            for _, instances in list(id_to_instances.items())[: self.adapter_spec.max_eval_instances]:
                eval_instances.extend(instances)

        hlog(
            f"{len(instances)} instances, "
            f"choosing {self.adapter_spec.max_train_instances}/{len(all_train_instances)} train instances, "
            f"{len(eval_instances)} eval instances"
        )

        # Accumulate all the request states due to adaptation
        request_states: List[RequestState] = []

        if self.adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            # Use the LM-specific method to adapt LM scenarios
            request_states = self.adapt_language_modeling(eval_instances)
        else:
            for train_trial_index in range(self.adapter_spec.num_train_trials):
                # Choose a random set of training instances
                random.seed(train_trial_index)
                train_instances = random.sample(
                    all_train_instances, min(len(all_train_instances), self.adapter_spec.max_train_instances)
                )

                # Create request_states
                for eval_index, eval_instance in enumerate(eval_instances):
                    prompt: str = self.construct_prompt(train_instances, eval_instance)

                    # Just print one prompt (useful for debugging)
                    if train_trial_index == 0 and eval_index == 0:
                        with htrack_block("Sample prompt"):
                            for line in prompt.split("\n"):
                                hlog(line)

                    # Define the request
                    method = self.adapter_spec.method

                    if method == ADAPT_GENERATION:
                        output_mapping = None
                        request = Request(
                            model=self.adapter_spec.model,
                            prompt=prompt,
                            num_completions=self.adapter_spec.num_outputs,
                            temperature=self.adapter_spec.temperature,
                            max_tokens=self.adapter_spec.max_tokens,
                            stop_sequences=self.adapter_spec.stop_sequences,
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

        Fits the prompt within the context window by removing in-context training examples.
        """

        def construct_prompt_helper(train_instances: List[Instance]) -> str:
            # Instructions
            blocks = []

            if self.adapter_spec.instructions:
                blocks.append(self.adapter_spec.instructions)

            # In-context training instances
            for instance in train_instances:
                blocks.append(self.construct_example_prompt(instance, include_output=True))

            blocks.append(self.construct_example_prompt(eval_instance, include_output=False))

            return "\n\n".join(blocks)

        orig_train_instances_count: int = len(train_instances)
        prompt: str = construct_prompt_helper(train_instances)

        # Following what was done for MMLU (https://arxiv.org/abs/2009.03300) to handle prompts that
        # exceed the max context length (https://github.com/hendrycks/test/blob/master/evaluate.py#L58),
        # we remove train instances one by one until it fits within the context window or
        # until we run out of train instances to remove.
        while (
            not self.tokenizer.fits_within_context_window(
                text=prompt, expected_completion_token_length=self.adapter_spec.max_tokens,
            )
            and len(train_instances) > 0
        ):
            train_instances = train_instances[:-1]
            prompt = construct_prompt_helper(train_instances)

        removed_train_instances_count: int = orig_train_instances_count - len(train_instances)
        if removed_train_instances_count > 0:
            hlog(
                f"The original constructed prompt exceeded the max context length. Removed "
                f"{removed_train_instances_count} in-context examples to fit it within the context window."
            )

        # If removing the in-context example is still not enough, we simply truncate the prompt.
        # Following the default truncation strategy used by HuggingFace, we truncate the text from the right.
        return self.tokenizer.truncate_from_right(prompt)

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

    def construct_language_modeling_prompt(
        self, conditioning_tokens: List[int], pred_tokens: List[int], tokenizer: Tokenizer, max_seq_len: int
    ) -> Tuple[str, int]:
        """
        Some subwords/symbols might translate to multiple tokens. e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"].

        When a subword of this type happens to be the last token of a chunk, we need to strip the leading and
        trailing bytes to ensure the prompt is a valid string.

        Since some tokens are removed, we also need to recompute num_conditioning_tokens.
        """
        raw_prompt: str = tokenizer.decode(conditioning_tokens + pred_tokens)
        prompt: str = raw_prompt.strip("\ufffd")
        num_leading_byte_tokens: int = max_seq_len + 1 - len(tokenizer.encode(raw_prompt.lstrip("\ufffd")))
        num_trailing_byte_tokens: int = max_seq_len + 1 - len(tokenizer.encode(raw_prompt.rstrip("\ufffd")))

        # There are no string tokens to predict
        if num_trailing_byte_tokens >= len(pred_tokens):
            num_conditioning_tokens = len(tokenizer.encode(prompt))
        # There are no conditioning string tokens
        elif num_leading_byte_tokens >= len(conditioning_tokens):
            num_conditioning_tokens = 1
        else:
            num_conditioning_tokens = len(conditioning_tokens) - num_leading_byte_tokens
        return prompt, num_conditioning_tokens

    def adapt_language_modeling(self, instances: List[Instance]) -> List[RequestState]:
        """ Code is adapted from:

        https://github.com/EleutherAI/lm_perplexity/blob/main/lm_perplexity/utils.py
        """
        request_states: List[RequestState] = []

        # TODO: Support other models and tokenizers
        assert self.adapter_spec.model.startswith("openai/")

        max_seq_len: int = self.tokenizer.max_sequence_length
        prefix_token: str = self.tokenizer.end_of_text_token

        for instance in instances:
            tokens = self.tokenizer.encode(instance.input)
            assert self.tokenizer.decode(tokens) == instance.input

            num_predicted_tokens = 0

            # Special handling for first window: predict all tokens
            # Raw token sequence format: [<str_tok1>, <str_tok2>, ..., <byte_tok1>, ...] (total length <= max_seq_len)
            # Convert it to: [<eot>, <str_tok1>, <str_tok2>, ...] (total length <= max_seq_len+1)
            # Num_conditioning_tokens = 1
            # Example: ["Hello", " world", "bytes:\xe2\x80"] => "<eot>Hello world"
            #
            # Note: There are trailing byte tokens in the raw sequence because some subwords/symbols might translate to
            # multiple tokens (e.g. ’ => ["bytes:\xe2\x80", "bytes:\x99"]) and we chunk documents by token, not by word.
            first_seq_len = min(max_seq_len, len(tokens))
            prompt = self.tokenizer.decode(self.tokenizer.encode(prefix_token) + tokens[:first_seq_len]).rstrip(
                "\ufffd"
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
                train_trial_index=0,
                output_mapping=None,
                request=request,
                result=None,
                num_conditioning_tokens=1,
            )
            request_states.append(request_state)
            num_predicted_tokens += first_seq_len

            while num_predicted_tokens < len(tokens):
                # Raw token sequence format:
                # [<cond_byte1>, ..., <cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, ..., <pred_byte1>, ...]
                # (total length <= max_seq_len+1)
                #
                # Convert it to: [<cond_str_tok1>, <cond_str_tok2>, ..., <pred_str_tok1>, <pred_str_tok2>. ...]
                # (total length <= max_seq_len+1)
                #
                # Example: conditioning_tokens=["bytes:\x99", "Exc"], pred_tokens=["use", " me", "bytes:\xe2\x80"] =>
                # prompt="Excuse me", num_conditioning_tokens = 1
                window_pred_len = min(len(tokens) - num_predicted_tokens, max_seq_len)
                window_end = num_predicted_tokens + window_pred_len
                conditioning_tokens = tokens[window_end - max_seq_len - 1 : num_predicted_tokens]
                pred_tokens = tokens[num_predicted_tokens:window_end]
                prompt, num_conditioning_tokens = self.construct_language_modeling_prompt(
                    conditioning_tokens, pred_tokens, self.tokenizer, max_seq_len
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
                    train_trial_index=0,
                    output_mapping=None,
                    request=request,
                    result=None,
                    num_conditioning_tokens=num_conditioning_tokens,
                )
                request_states.append(request_state)
                num_predicted_tokens += window_pred_len

        return request_states
