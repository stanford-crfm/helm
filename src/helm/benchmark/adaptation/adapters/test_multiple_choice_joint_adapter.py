# mypy: check_untyped_defs = False
from helm.benchmark.scenarios.scenario import TEST_SPLIT, TRAIN_SPLIT, Instance, Input, Output, Reference, CORRECT_TAG
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .adapter_factory import AdapterFactory, ADAPT_MULTIPLE_CHOICE_JOINT
from .test_adapter import TestAdapter


class TestMultipleChoiceJointAdapter(TestAdapter):
    def test_sample_examples(self):
        adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE_JOINT, model="openai/ada", max_train_instances=4)
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
        adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE_JOINT, model="openai/ada", max_train_instances=2)
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        examples = adapter.sample_examples(all_train_instances=[], seed=0)
        assert len(examples) == 0

    def test_sample_examples_greater_max_train_instances(self):
        adapter_spec = AdapterSpec(method=ADAPT_MULTIPLE_CHOICE_JOINT, model="openai/ada", max_train_instances=10)
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        all_train_instances = [
            Instance(Input(text="say no"), references=[Reference(Output(text="no"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
            Instance(Input(text="say yes"), references=[Reference(Output(text="yes"), tags=[CORRECT_TAG])]),
        ]

        examples = adapter.sample_examples(all_train_instances, seed=0)
        assert len(examples) == 3

    def test_multiple_correct_reference(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT, model="openai/ada", max_train_instances=10, sample_train=False
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
        actual_instances = adapter.adapt(train_instances + [eval_instance], parallelism=1).request_states
        assert len(actual_instances) == 1
        assert actual_instances[0].request.prompt == (
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
            model="openai/ada",
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
        actual_instances = adapter.adapt(train_instances + [eval_instance], parallelism=1).request_states
        assert len(actual_instances) == 1
        assert actual_instances[0].request.prompt == (
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
