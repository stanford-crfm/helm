from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_RANKING_BINARY,
    AdapterSpec,
)


def format_instructions(instructions: str) -> str:
    if len(instructions) > 0:
        instructions += "\n"
    return instructions


def get_multiple_choice_joint_adapter_spec(
    instructions: str,
    input_noun: Optional[str],
    output_noun: str,
    num_outputs: int = 5,
    max_train_instances: int = 5,
    max_tokens: int = 5,
    sample_train: bool = True,
    **kwargs,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]: [output]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]:
    """

    input_prefix = kwargs.pop("input_prefix", f"{input_noun}: " if input_noun is not None else "")
    input_suffix = kwargs.pop("input_suffix", "\n" if input_noun is not None else "")
    output_prefix = kwargs.pop("output_prefix", f"{output_noun}: ")
    output_suffix = kwargs.pop("output_suffix", "\n")

    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=["\n"],
        sample_train=sample_train,
        **kwargs,
    )


def get_multiple_choice_joint_chain_of_thought_adapter_spec(
    instructions: str,
    input_noun: Optional[str],
    output_noun: str,
    num_outputs: int = 5,
    max_train_instances: int = 5,
    max_tokens: int = 5,
    sample_train: bool = True,
    **kwargs,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]: [output]

    [input_noun]: [input]
    [reference_1]
    ...
    [reference_k]
    [output_noun]:
    """

    input_prefix = kwargs.pop("input_prefix", f"{input_noun}: " if input_noun is not None else "")
    input_suffix = kwargs.pop("input_suffix", "\n" if input_noun is not None else "")
    output_prefix = kwargs.pop("output_prefix", f"{output_noun}: ")
    output_suffix = kwargs.pop("output_suffix", "\n")

    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=["\n"],
        sample_train=sample_train,
        **kwargs,
    )


def get_multiple_choice_separate_adapter_spec(method: str, empty_input: bool = False) -> AdapterSpec:
    """
    [input] [reference_i]
    or
    [reference_i]
    """
    assert method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}

    return AdapterSpec(
        method=method,
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix=" " if not empty_input else "",
        output_suffix="",
        # Separate is basically language modeling, so can't easily use in-context examples
        max_train_instances=0,
        num_outputs=1,
        max_tokens=0,
        temperature=0.0,
    )


def get_multiple_choice_adapter_spec(
    method: str,
    instructions: str,
    input_noun: Optional[str],
    output_noun: str,
    max_train_instances: int = 5,
    num_outputs: int = 5,
    max_tokens: int = 1,
    empty_input: bool = False,
    sample_train: bool = True,
    **kwargs,
):
    """
    Toggle between joint and separate adapters.
    """
    if method == ADAPT_MULTIPLE_CHOICE_JOINT:
        return get_multiple_choice_joint_adapter_spec(
            instructions,
            input_noun,
            output_noun,
            max_train_instances=max_train_instances,
            num_outputs=num_outputs,
            max_tokens=max_tokens,
            sample_train=sample_train,
            **kwargs,
        )
    elif method == ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT:
        return get_multiple_choice_joint_chain_of_thought_adapter_spec(
            instructions,
            input_noun,
            output_noun,
            max_train_instances=max_train_instances,
            num_outputs=num_outputs,
            max_tokens=max_tokens,
            sample_train=sample_train,
            **kwargs,
        )
    elif method in {ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED}:
        return get_multiple_choice_separate_adapter_spec(method, empty_input)
    else:
        raise ValueError(f"Invalid adaptation method: {method}")


def get_ranking_binary_adapter_spec(
    instructions: str = "",
    document_noun: str = "Passage",
    query_noun: str = "Query",
    output_prefix: str = "Does the passage answer the query?",
    output_noun: str = "Answer",
    max_train_instances: int = 4,
    num_outputs: int = 1,
    num_train_trials: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 5,
    **kwargs,
) -> AdapterSpec:
    """
    [instructions]

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]

    ...

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]

    [object_noun]: [object]
    [query_noun]: [query]
    [prompt_noun]: [prompt_content]
    [output_noun]: [output]
    """
    msg = (
        "There must be an even number of in-context examples to ensure that"
        "an equal number of positive and negative examples are included."
    )
    assert max_train_instances % 2 == 0, msg
    max_train_instances = int(max_train_instances / 2)

    return AdapterSpec(
        method=ADAPT_RANKING_BINARY,
        instructions=format_instructions(instructions),
        input_prefix=f"{query_noun}: ",
        input_suffix="\n",
        reference_prefix=f"{document_noun}: ",
        reference_suffix="\n",
        output_prefix=f"{output_prefix}\n{output_noun}: ",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        num_train_trials=num_train_trials,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_completion_adapter_spec(
    instructions: str = "",
    input_prefix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_train_instances: int = 0,
    temperature: float = 0.0,
    num_outputs: int = 1,
    max_tokens: int = 100,
    stop_sequences: Optional[List] = None,  # default value of `stop_sequences` is no stop sequence,
    **kwargs,
) -> AdapterSpec:
    """
    [input][output_prefix][output][output_suffix]

    [input][output_prefix]
    """
    if stop_sequences is None:
        stop_sequences = []

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix,
        input_suffix="",
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        max_train_instances=max_train_instances,
        temperature=temperature,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        **kwargs,
    )


def get_generation_adapter_spec(
    instructions: str = "",
    input_noun: Optional[str] = None,
    newline_after_input_noun: bool = False,
    output_noun: Optional[str] = None,
    newline_after_output_noun: bool = False,
    max_train_instances: int = 5,
    num_outputs: int = 1,
    max_tokens: int = 5,
    stop_sequences: Optional[List] = None,  # default value of `stop_sequences` is ["\n"]
    temperature: float = 0.0,
    multi_label: bool = False,
    sample_train: bool = True,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [output_noun]: [output]

    [input_noun]: [input]
    [output_noun]:
    """

    def format_prefix(noun: Optional[str], append_new_line: bool) -> str:
        """
        When `append_new_line` is False:
            [input_noun]: [input]

        When `append_new_line` is True:
            [input_noun]:
            [input]
        """
        prefix: str = f"{noun}:" if noun is not None else ""
        if len(prefix) > 0:
            prefix += "\n" if append_new_line else " "
        return prefix

    if stop_sequences is None:
        stop_sequences = ["\n"]

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=format_instructions(instructions),
        input_prefix=format_prefix(input_noun, append_new_line=newline_after_input_noun),
        input_suffix="\n",
        output_prefix=format_prefix(output_noun, append_new_line=newline_after_output_noun),
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        multi_label=multi_label,
        sample_train=sample_train,
    )


def get_instruct_adapter_spec(
    num_outputs: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AdapterSpec:
    """
    Zero-shot instruction-following.
    """
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=[],
    )


def get_few_shot_instruct_adapter_spec(
    num_outputs: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    max_train_instances: int = 0,
) -> AdapterSpec:
    """
    Few-shot instruction-following.
    """
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=[],
    )


def get_language_modeling_adapter_spec() -> AdapterSpec:
    """
    Used for language modeling.
    """
    return AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING,
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=0,
        temperature=0.0,
    )


def get_summarization_adapter_spec(num_sents: Optional[int], max_train_instances: int = 5, **kwargs) -> AdapterSpec:
    """
    Used for summarization.
    """

    if num_sents == 1:
        out_pref = "Summarize the above article in 1 sentence.\n"
    elif num_sents is None:
        out_pref = "Summarize the above article.\n"
    else:
        out_pref = f"Summarize the above article in {num_sents} sentences.\n"

    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="###\nArticle: ",
        input_suffix="\n\n",
        output_prefix=out_pref,
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        stop_sequences=["###"],  # Separator between few-shot instances.
        **kwargs,
    )


def get_machine_translation_adapter_spec(
    source_language, target_language, max_train_instances, **kwargs
) -> AdapterSpec:
    """
    Used for machine translation.
    """
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=f"Translate the following sentences from {source_language} to {target_language}.",
        input_prefix=f"{source_language}: ",
        input_suffix="\n",
        output_prefix=f"{target_language}: ",
        output_suffix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        stop_sequences=["\n\n"],
        temperature=0.0,
        **kwargs,
    )
