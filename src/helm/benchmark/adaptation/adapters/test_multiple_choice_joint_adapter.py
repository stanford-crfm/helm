from helm.benchmark.scenarios.scenario import Instance, Input, Output, Reference, CORRECT_TAG
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
