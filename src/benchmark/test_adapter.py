from typing import List

from .scenario import CORRECT_TAG, create_scenario, Instance, Reference
from .run_specs import get_scenario_spec1, get_adapter_spec1, get_adapter_spec1_with_data_augmentation
from .adapter import ADAPT_GENERATION, Adapter, AdapterSpec


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
    perturbation_tag: str = "extra_space|num_spaces=5"
    assert perturbation_tag in adapter_spec.data_augmenter_spec.tags

    # After adaptation, check that the augmentation has been applied
    # by verifying that the instances with the augmentation tag are perturbed
    scenario_state = Adapter(adapter_spec).adapt(scenario)
    for instance in scenario_state.instances:
        if perturbation_tag in instance.tags:
            assert " " * 5 in instance.input
        else:
            assert " " * 5 not in instance.input


def test_construct_prompt():
    adapter_spec = AdapterSpec(
        model="openai/davinci", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
    )
    adapter = Adapter(adapter_spec)
    correct_reference = Reference(output="", tags=[CORRECT_TAG])
    train_instances: List[Instance] = [
        Instance(input="train", references=[correct_reference], tags=[]) for _ in range(2049)
    ]
    eval_instances = Instance(input="eval", references=[], tags=[])
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
