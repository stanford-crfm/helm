# mypy: check_untyped_defs = False
from typing import List, Set
from helm.benchmark.scenarios.scenario import TEST_SPLIT, TRAIN_SPLIT, Instance, Input, Output, Reference, CORRECT_TAG
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory, ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.adapters.test_adapter import TestAdapter


def _make_instance(
    text: str, reference_texts: List[str], correct_references: Set[int], is_eval: bool = False
) -> Instance:
    references = []
    for i, reference_text in enumerate(reference_texts):
        tags = [CORRECT_TAG] if i in correct_references else []
        references.append(Reference(Output(text=reference_text), tags=tags))

    split = TEST_SPLIT if is_eval else TRAIN_SPLIT
    return Instance(Input(text=text), references=references, split=split)


class TestMultipleChoiceJointAdapter(TestAdapter):
    def test_sample_examples(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=4,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            Instance(Input(text="say no"), references=[Reference(Output(text="no"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes1"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes2"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes3"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes4"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
        ]

        examples = adapter.sample_examples(all_train_instances, seed=0)
        assert len(examples) == 4

        # An instance with "say yes" should have be sampled first before "say no"
        assert examples[0].input.text == "say yes4"
        assert examples[1].input.text == "say no"
        assert examples[2].input.text == "say yes1"
        assert examples[3].input.text == "say yes3"

    def test_sample_examples_no_train_instances(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=2,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        examples = adapter.sample_examples(all_train_instances=[], seed=0)
        assert len(examples) == 0

    def test_sample_examples_greater_max_train_instances(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=10,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            Instance(Input(text="say no"), references=[Reference(Output(text="no"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
        ]

        examples = adapter.sample_examples(all_train_instances, seed=0)
        assert len(examples) == 3

    def test_sample_examples_unique_labels(self):
        """This is a demonstration of behavior reported in issue #2224."""
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=3,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            # Three with 0 being correct.
            _make_instance("one", ["0", "1"], correct_references={0}),
            _make_instance("two", ["2", "3"], correct_references={0}),
            _make_instance("three", ["4", "5"], correct_references={0}),
            # Two with 1 being correct.
            _make_instance("four", ["6", "7"], correct_references={1}),
            _make_instance("five", ["8", "9"], correct_references={1}),
        ]
        eval_instance = _make_instance("eval", ["10", "11"], correct_references={1}, is_eval=True)
        request_states = adapter.adapt(all_train_instances + [eval_instance], parallelism=1)
        assert len(request_states) == 1
        # In every case, we are showing that model that Output should be "A".
        assert request_states[0].request.prompt == (
            "Input: three\n"
            "A. 4\n"
            "B. 5\n"
            "Output: A\n"
            "\n"
            "Input: two\n"
            "A. 2\n"
            "B. 3\n"
            "Output: A\n"
            "\n"
            "Input: one\n"
            "A. 0\n"
            "B. 1\n"
            "Output: A\n"
            "\n"
            "Input: eval\n"
            "A. 10\n"
            "B. 11\n"
            "Output:"
        )

    def test_multiple_correct_reference(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=10,
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
        request_states = adapter.adapt(train_instances + [eval_instance], parallelism=1)
        assert len(request_states) == 1
        assert request_states[0].request.prompt == (
            "Input: Second reference is correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output: B\n\n"
            "Input: First and second references are correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output: A\n\n"
            "Input: First reference is correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output:"
        )

    def test_multiple_correct_reference_multi_label(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=10,
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
        request_states = adapter.adapt(train_instances + [eval_instance], parallelism=1)
        assert len(request_states) == 1
        assert request_states[0].request.prompt == (
            "Input: Second reference is correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output: B\n\n"
            "Input: First and second references are correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output: A, B\n\n"
            "Input: First reference is correct\n"
            "A. First\n"
            "B. Second\n"
            "C. Third\n"
            "Output:"
        )

    def test_reference_prefix(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT,
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            max_train_instances=10,
            sample_train=False,
            reference_prefix="  1: ",
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
                Input(text="Third reference is correct"),
                references=[
                    Reference(Output(text="First"), tags=[]),
                    Reference(Output(text="Second"), tags=[]),
                    Reference(Output(text="Third"), tags=[CORRECT_TAG]),
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
        request_states = adapter.adapt(train_instances + [eval_instance], parallelism=1)
        assert len(request_states) == 1
        assert request_states[0].request.prompt == (
            "Input: Second reference is correct\n"
            "  1: First\n"
            "  2: Second\n"
            "  3: Third\n"
            "Output: 2\n\n"
            "Input: Third reference is correct\n"
            "  1: First\n"
            "  2: Second\n"
            "  3: Third\n"
            "Output: 3\n\n"
            "Input: First reference is correct\n"
            "  1: First\n"
            "  2: Second\n"
            "  3: Third\n"
            "Output:"
        )
