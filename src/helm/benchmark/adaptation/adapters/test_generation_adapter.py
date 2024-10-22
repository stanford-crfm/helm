# mypy: check_untyped_defs = False
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    create_scenario,
    Instance,
    Reference,
    Input,
    Output,
)
from helm.benchmark.run_specs.simple_run_specs import get_simple1_spec
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory, ADAPT_GENERATION
from helm.benchmark.adaptation.adapters.generation_adapter import GenerationAdapter
from helm.benchmark.adaptation.adapters.test_adapter import TestAdapter


class TestGenerationAdapter(TestAdapter):
    def test_adapt(self):
        run_spec = get_simple1_spec()
        scenario = create_scenario(run_spec.scenario_spec)
        adapter_spec = run_spec.adapter_spec
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        instances = scenario.get_instances(output_path="")
        request_states = adapter.adapt(instances, parallelism=1)
        non_train_instances = [instance for instance in instances if instance.split != TRAIN_SPLIT]

        # Make sure we generated the right number of request_states:
        # For each trial, instance and reference (+ 1 for free-form generation).
        assert len(non_train_instances) * adapter_spec.num_train_trials == len(request_states)

    def test_construct_prompt(self):
        adapter_spec = AdapterSpec(
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
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
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            method=ADAPT_GENERATION,
            input_prefix="",
            output_prefix="",
            max_tokens=100,
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
        assert prompt_text.count("eval") == 924

    def test_sample_examples_without_references(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION, model="openai/gpt2", model_deployment="huggingface/gpt2", max_train_instances=1
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            Instance(Input(text="prompt1"), references=[]),
            Instance(Input(text="prompt2"), references=[]),
            Instance(Input(text="prompt3"), references=[]),
        ]

        examples = adapter.sample_examples(all_train_instances, seed=0)
        assert len(examples) == 1

    def test_sample_examples_open_ended_generation(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION, model="openai/gpt2", model_deployment="huggingface/gpt2", max_train_instances=3
        )
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
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION, model="openai/gpt2", model_deployment="huggingface/gpt2", max_train_instances=5
        )
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

    def test_multiple_correct_reference(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=2,
            sample_train=False,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        train_instances = [
            Instance(
                Input(text="Second reference is correct"),
                references=[
                    Reference(Output(text="First"), tags=[]),
                    Reference(Output(text="Second"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Third"), tags=[]),
                ],
                split=TRAIN_SPLIT,
            ),
            Instance(
                Input(text="First and second references are correct"),
                references=[
                    Reference(Output(text="First"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Second"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Third"), tags=[]),
                ],
                split=TRAIN_SPLIT,
            ),
        ]
        eval_instance = Instance(
            Input(text="First reference is correct"),
            references=[
                Reference(Output(text="First"), tags=[CORRECT_TAG]),
                Reference(Output(text="Second"), tags=[]),
                Reference(Output(text="Third"), tags=[]),
            ],
            split=TEST_SPLIT,
        )
        actual_instances = adapter.adapt(train_instances + [eval_instance], parallelism=1)
        assert len(actual_instances) == 1
        assert actual_instances[0].request.prompt == (
            "Input: Second reference is correct\n"
            "Output: Second\n\n"
            "Input: First and second references are correct\n"
            "Output: First\n\n"
            "Input: First reference is correct\n"
            "Output:"
        )

    def test_multiple_correct_reference_multi_label(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_GENERATION,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=2,
            multi_label=True,
            sample_train=False,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        train_instances = [
            Instance(
                Input(text="Second reference is correct"),
                references=[
                    Reference(Output(text="First"), tags=[]),
                    Reference(Output(text="Second"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Third"), tags=[]),
                ],
                split=TRAIN_SPLIT,
            ),
            Instance(
                Input(text="First and second references are correct"),
                references=[
                    Reference(Output(text="First"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Second"), tags=[CORRECT_TAG]),
                    Reference(Output(text="Third"), tags=[]),
                ],
                split=TRAIN_SPLIT,
            ),
        ]
        eval_instance = Instance(
            Input(text="First reference is correct"),
            references=[
                Reference(Output(text="First"), tags=[CORRECT_TAG]),
                Reference(Output(text="Second"), tags=[]),
                Reference(Output(text="Third"), tags=[]),
            ],
            split=TEST_SPLIT,
        )
        actual_instances = adapter.adapt(train_instances + [eval_instance], parallelism=1)
        assert len(actual_instances) == 1
        assert actual_instances[0].request.prompt == (
            "Input: Second reference is correct\n"
            "Output: Second\n\n"
            "Input: First and second references are correct\n"
            "Output: First, Second\n\n"
            "Input: First reference is correct\n"
            "Output:"
        )

    def test_construct_prompt_image_generation(self):
        adapter_spec = AdapterSpec(
            model_deployment="openai/dall-e-2",
            method=ADAPT_GENERATION,
            input_prefix="",
            input_suffix="",
            output_prefix="",
            output_suffix="",
            max_train_instances=0,
            num_outputs=1,
            max_tokens=0,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        assert isinstance(adapter, GenerationAdapter)

        eval_instance = Instance(Input(text="a blue dog"), references=[])
        prompt: Prompt = adapter.construct_prompt([], eval_instance, include_output=False, reference_index=None)

        assert adapter.window_service.fits_within_context_window(prompt.text)
        assert prompt.text == "a blue dog"
