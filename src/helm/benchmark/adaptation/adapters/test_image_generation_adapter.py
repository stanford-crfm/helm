from helm.benchmark.scenarios.scenario import Instance, Input
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .adapter_factory import AdapterFactory, ADAPT_GENERATION
from .test_adapter import TestAdapter
from .image_generation_adapter import ImageGenerationAdapter


class TestImageGenerationAdapter(TestAdapter):
    def test_construct_prompt(self):
        adapter_spec = AdapterSpec(
            model_deployment="openai/dalle-2",
            method=ADAPT_GENERATION,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        assert isinstance(adapter, ImageGenerationAdapter)

        eval_instance = Instance(Input(text="a blue dog"), references=[])
        prompt: Prompt = adapter.construct_prompt([], eval_instance, include_output=False, reference_index=None)

        assert adapter.window_service.fits_within_context_window(prompt.text)
        assert prompt.text == "a blue dog"
