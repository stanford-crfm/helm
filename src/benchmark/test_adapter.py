from typing import List

from .scenario import CORRECT_TAG, create_scenario, Instance, Reference
from .run_specs import (
    get_scenario_spec1,
    get_empty_scenario_spec,
    get_adapter_spec1,
    get_adapter_spec1_with_data_augmentation,
    get_simple_interactions_adapter_spec,
)
from .adapter import ADAPT_GENERATION, ADAPT_LANGUAGE_MODELING, Adapter, AdapterSpec
from proxy.tokenizer.openai_token_counter import OpenAITokenCounter


def test_adapter1():
    scenario = create_scenario(get_scenario_spec1())
    adapter_spec = get_adapter_spec1()

    scenario_state = Adapter(adapter_spec).adapt(scenario)

    # Make sure we generated the right number of request_states:
    # For each trial, instance and reference (+ 1 for free-form generation).
    num_instances = len(scenario_state.instances)
    assert num_instances * adapter_spec.num_train_trials == len(scenario_state.request_states)

    # TODO: write tests to check the contents of the actual prompt
    #       https://github.com/stanford-crfm/benchmarking/issues/50


def test_adapter1_with_data_augmentation():
    scenario = create_scenario(get_scenario_spec1())
    adapter_spec = get_adapter_spec1_with_data_augmentation()

    # After adaptation, check that the data augmentation has been applied
    # by verifying that the instances with the perturbation tag are perturbed
    scenario_state = Adapter(adapter_spec).adapt(scenario)
    for instance in scenario_state.instances:
        if instance.perturbation.name == "extra_space":
            assert " " * 5 in instance.input
        else:
            assert " " * 5 not in instance.input


def test_adapter_with_interaction_mode():
    scenario_state = Adapter(adapter_spec=get_simple_interactions_adapter_spec()).adapt(
        scenario=create_scenario(get_empty_scenario_spec())
    )
    assert scenario_state.interaction_outcomes[0].conversation == ["Hi", "Hello", "Bye", "Goodbye"]


def test_construct_prompt():
    adapter_spec = AdapterSpec(
        model="openai/davinci", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
    )
    adapter = Adapter(adapter_spec)
    correct_reference = Reference(output="", tags=[CORRECT_TAG])
    train_instances: List[Instance] = [Instance(input="train", references=[correct_reference]) for _ in range(2049)]
    eval_instances = Instance(input="eval", references=[])
    prompt: str = adapter.construct_prompt(train_instances, eval_instances)

    # Ensure the prompt fits within the context window
    assert adapter.token_counter.fits_within_context_window(adapter_spec.model, prompt)

    # Ensure the in-context examples were removed before touching the evaluation instance
    assert prompt.endswith("eval")

    # Using a Jurassic model should yield the same prompt as using a GPT-3 model,
    # since we use a GPT-2 tokenizer to truncate prompts for both.
    adapter_spec = AdapterSpec(
        model="ai21/j1-jumbo", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
    )
    adapter = Adapter(adapter_spec)
    assert adapter.construct_prompt(train_instances, eval_instances) == prompt


def test_construct_language_modeling_prompt():
    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING, input_prefix="", model="openai/davinci", output_prefix="", max_tokens=0,
    )
    adapter = Adapter(adapter_spec)
    tokenizer = OpenAITokenCounter().tokenizer

    # The tokens translate to: '�Excuse me�'
    conditioning_tokens, pred_tokens = [110, 40127], [1904, 502, 447]
    prompt, num_conditioning_tokens = adapter.construct_language_modeling_prompt(
        conditioning_tokens, pred_tokens, tokenizer, 5
    )

    # Ensure the prompt is correct
    assert prompt == "Excuse me"

    # Ensure the number of conditioning tokens is correct
    assert num_conditioning_tokens == 1
