from typing import List

from .adapter_service import AdapterService
from .scenario import CORRECT_TAG, create_scenario, Instance, Reference
from .run_specs import get_scenario_spec1, get_adapter_spec1
from .adapter import ADAPT_GENERATION, ADAPT_LANGUAGE_MODELING, ADAPT_MULTIPLE_CHOICE, Adapter, AdapterSpec
from proxy.remote_service import RemoteService
from proxy.tokenizer.tokenizer_factory import TokenizerFactory
from common.authentication import Authentication


def get_test_adapter_service() -> AdapterService:
    return AdapterService(RemoteService("test"), Authentication("test"))


def test_adapter1():
    scenario = create_scenario(get_scenario_spec1())
    adapter_spec = get_adapter_spec1()
    scenario_state = Adapter(adapter_spec, get_test_adapter_service()).adapt(scenario.get_instances())

    # Make sure we generated the right number of request_states:
    # For each trial, instance and reference (+ 1 for free-form generation).
    num_instances = len(scenario_state.instances)
    assert num_instances * adapter_spec.num_train_trials == len(scenario_state.request_states)

    # TODO: write tests to check the contents of the actual prompt
    #       https://github.com/stanford-crfm/benchmarking/issues/50


def test_construct_prompt():
    adapter_spec = AdapterSpec(
        model="openai/davinci", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
    )
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    correct_reference = Reference(output="", tags=[CORRECT_TAG])
    train_instances: List[Instance] = [Instance(input="train", references=[correct_reference]) for _ in range(2049)]
    eval_instances = Instance(input="eval", references=[])
    prompt: str = adapter.construct_prompt(train_instances, eval_instances)

    # Ensure the prompt fits within the context window
    assert adapter.tokenizer.fits_within_context_window(prompt)

    # Ensure the in-context examples were removed before touching the evaluation instance
    assert prompt.endswith("eval")


def test_construct_prompt_with_truncation():
    adapter_spec = AdapterSpec(
        model="openai/davinci", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
    )
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    correct_reference = Reference(output="", tags=[CORRECT_TAG])
    train_instances: List[Instance] = [Instance(input="train", references=[correct_reference]) for _ in range(100)]
    eval_instances = Instance(input="eval" * 2049, references=[])
    prompt: str = adapter.construct_prompt(train_instances, eval_instances)

    # Ensure the prompt fits within the context window
    assert adapter.tokenizer.fits_within_context_window(prompt)

    # Ensure that all the in-context examples were completely removed and we had to truncate the eval Instance input
    assert "train" not in prompt
    assert prompt.count("eval") == 1948


def test_construct_language_modeling_prompt():
    model: str = "openai/davinci"
    adapter_spec = AdapterSpec(
        method=ADAPT_LANGUAGE_MODELING, input_prefix="", model=model, output_prefix="", max_tokens=0,
    )
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    tokenizer = TokenizerFactory.get_tokenizer(model)

    # The tokens translate to: '�Excuse me�'
    conditioning_tokens, pred_tokens = [110, 40127], [1904, 502, 447]
    prompt, num_conditioning_tokens = adapter.construct_language_modeling_prompt(
        conditioning_tokens, pred_tokens, tokenizer, 5, ""
    )

    # Ensure the prompt is correct
    assert prompt == "Excuse me"

    # Ensure the number of conditioning tokens is correct
    assert num_conditioning_tokens == 1


def test_sample_examples():
    adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE, max_train_instances=4)
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    all_train_instances = [
        Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])]),
        Instance("say yes1", references=[Reference("yes", tags=[CORRECT_TAG])]),
        Instance("say yes2", references=[Reference("yes", tags=[CORRECT_TAG])]),
        Instance("say yes3", references=[Reference("yes", tags=[CORRECT_TAG])]),
        Instance("say yes4", references=[Reference("yes", tags=[CORRECT_TAG])]),
    ]

    examples = adapter.sample_examples(all_train_instances, seed=0)
    assert len(examples) == 4

    # An instance with "say yes" should have be sampled first before "say no"
    assert examples[0].input == "say yes4"
    assert examples[1].input == "say no"
    assert examples[2].input == "say yes1"
    assert examples[3].input == "say yes3"


def test_sample_examples_no_train_instances():
    adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE, max_train_instances=2)
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    examples = adapter.sample_examples(all_train_instances=[], seed=0)
    assert len(examples) == 0


def test_sample_examples_greater_max_train_instances():
    adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE, max_train_instances=10)
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    all_train_instances = [
        Instance("say no", references=[Reference("no", tags=[CORRECT_TAG])]),
        Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),
        Instance("say yes", references=[Reference("yes", tags=[CORRECT_TAG])]),
    ]

    examples = adapter.sample_examples(all_train_instances, seed=0)
    assert len(examples) == 3


def test_sample_examples_without_references():
    adapter_spec = AdapterSpec(method=ADAPT_LANGUAGE_MODELING, max_train_instances=1)
    adapter = Adapter(adapter_spec, get_test_adapter_service())
    all_train_instances = [
        Instance("prompt1", references=[]),
        Instance("prompt2", references=[]),
        Instance("prompt3", references=[]),
    ]

    examples = adapter.sample_examples(all_train_instances, seed=0)
    assert len(examples) == 1
