from typing import List

from helm.benchmark.scenarios.scenario import CORRECT_TAG, create_scenario, Instance, Reference, Input, Output
from helm.benchmark.run_specs import get_scenario_spec1, get_adapter_spec1
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .adapter_factory import AdapterFactory, ADAPT_GENERATION
from .test_adapter import TestAdapter


class TestGenerationAdapter(TestAdapter):
    def test_adapt(self):
        scenario = create_scenario(get_scenario_spec1())
        adapter_spec = get_adapter_spec1()
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        scenario_state = adapter.adapt(scenario.get_instances(), parallelism=1)

        # Make sure we generated the right number of request_states:
        # For each trial, instance and reference (+ 1 for free-form generation).
        num_instances = len(scenario_state.instances)
        assert num_instances * adapter_spec.num_train_trials == len(scenario_state.request_states)

    def test_construct_prompt(self):
        adapter_spec = AdapterSpec(
            model="openai/davinci",
            method=ADAPT_GENERATION,
            input_prefix="",
            input_suffix="",
            output_prefix="",
            output_suffix="",
            max_tokens=100,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        correct_reference = Reference(Output(text=""), tags=[CORRECT_TAG])
        train_instances: List[Instance] = [
            Instance(Input(text="train"), references=[correct_reference]) for _ in range(2049)
        ]
        eval_instances = Instance(Input(text="eval"), references=[])
        prompt: Prompt = adapter.construct_prompt(
            train_instances, eval_instances, include_output=False, reference_index=None
        )
        prompt_text: str = prompt.text

        # Ensure the prompt fits within the context window
        assert adapter.window_service.fits_within_context_window(prompt_text)

        # Ensure the in-context examples were removed before touching the evaluation instance
        assert prompt_text.endswith("eval")

    def test_construct_prompt_with_truncation(self):
        adapter_spec = AdapterSpec(
            model="openai/davinci", method=ADAPT_GENERATION, input_prefix="", output_prefix="", max_tokens=100
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        correct_reference = Reference(Output(text=""), tags=[CORRECT_TAG])
        train_instances: List[Instance] = [
            Instance(Input(text="train"), references=[correct_reference]) for _ in range(100)
        ]
        eval_instances = Instance(Input(text="eval" * 2049), references=[])
        prompt: Prompt = adapter.construct_prompt(
            train_instances, eval_instances, include_output=False, reference_index=None
        )
        prompt_text: str = prompt.text

        # Ensure the prompt fits within the context window
        assert adapter.window_service.fits_within_context_window(prompt_text)

        # Ensure that all the in-context examples were completely removed and we had to truncate the eval Instance input
        assert "train" not in prompt_text
        assert prompt_text.count("eval") == 1948

    def test_sample_examples_without_references(self):
        adapter_spec = AdapterSpec(method=ADAPT_GENERATION, model="openai/ada", max_train_instances=1)
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            Instance(Input(text="prompt1"), references=[]),
            Instance(Input(text="prompt2"), references=[]),
            Instance(Input(text="prompt3"), references=[]),
        ]

        examples = adapter.sample_examples(all_train_instances, seed=0)
        assert len(examples) == 1

    def test_sample_examples_open_ended_generation(self):
        adapter_spec = AdapterSpec(method=ADAPT_GENERATION, model="openai/ada", max_train_instances=3)
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)

        all_train_instances: List[Instance] = [
            Instance(Input(text=f"prompt{i}"), references=[Reference(Output(text=f"reference{i}"), tags=[CORRECT_TAG])])
            for i in range(1, 10)
        ]
        seed0_examples: List[Instance] = adapter.sample_examples(all_train_instances, seed=0)
        seed1_examples: List[Instance] = adapter.sample_examples(all_train_instances, seed=1)

        assert len(seed0_examples) == len(seed1_examples) == 3
        assert seed0_examples != seed1_examples, "Examples should differ when changing the seed"

    def test_sample_examples_open_ended_generation_stress(self):
        adapter_spec = AdapterSpec(method=ADAPT_GENERATION, model="openai/ada", max_train_instances=5)
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)

        all_train_instances: List[Instance] = [
            Instance(Input(text="prompt3"), references=[Reference(Output(text="reference3"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt3"), references=[Reference(Output(text="reference3"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt1"), references=[Reference(Output(text="reference1"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt1"), references=[Reference(Output(text="reference1"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt1"), references=[Reference(Output(text="reference1"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt2"), references=[Reference(Output(text="reference2"), tags=[CORRECT_TAG])]),
            Instance(Input(text="prompt2"), references=[Reference(Output(text="reference2"), tags=[CORRECT_TAG])]),
        ]
        # Add prompt4,..,prompt100
        for i in range(4, 100):
            all_train_instances.append(
                Instance(
                    Input(text=f"prompt{i}"), references=[Reference(Output(text=f"reference{i}"), tags=[CORRECT_TAG])]
                )
            )

        previous_train_instances: List[List[Instance]] = []
        for seed in range(10):
            train_instances = adapter.sample_examples(all_train_instances, seed=seed)
            # Ensure calling the method with the same seed again picks the same train Instances
            assert train_instances == adapter.sample_examples(all_train_instances, seed=seed)

            assert len(train_instances) == 5
            assert train_instances[0].input.text == "prompt1", "prompt1 Instance had the most common label: reference1"
            assert train_instances[1].input.text in ["prompt2", "prompt3"]
            assert train_instances[2].input.text in ["prompt2", "prompt3"]
            assert train_instances[3].input.text not in ["prompt1", "prompt2", "prompt3"]
            assert train_instances[4].input.text not in ["prompt1", "prompt2", "prompt3"]

            # Ensure we haven't seen the same in-context examples before from previous seeds
            for other_train_instances in previous_train_instances:
                assert train_instances != other_train_instances, "Examples should differ when changing the seed"
            previous_train_instances.append(train_instances)
