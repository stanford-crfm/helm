from helm.benchmark.scenarios.scenario import Instance, Input
from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.adapter_spec import TextToImageAdapterSpec
from .adapter_factory import AdapterFactory, ADAPT_GENERATION
from .test_adapter import TestAdapter


class TestImageGenerationAdapter(TestAdapter):
    def test_construct_prompt(self):
        adapter_spec = TextToImageAdapterSpec(
            model="together/StableDiffusion",
            method=ADAPT_GENERATION,
            medium="painting",
            modifications=["8K", "photorealistic"],
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        eval_instance = Instance(Input(text="a blue dog"), references=[])
        prompt: Prompt = adapter.construct_prompt([], eval_instance, include_output=False, reference_index=None)

        assert adapter.window_service.fits_within_context_window(prompt.text)
        assert prompt.text == "a painting of a blue dog, 8K, photorealistic"

    def test_construct_prompt_no_medium(self):
        adapter_spec = TextToImageAdapterSpec(
            model="together/StableDiffusion", method=ADAPT_GENERATION, modifications=["8K"]
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        eval_instance = Instance(Input(text="a blue dog"), references=[])
        prompt: Prompt = adapter.construct_prompt([], eval_instance, include_output=False, reference_index=None)

        assert adapter.window_service.fits_within_context_window(prompt.text)
        assert prompt.text == "a blue dog, 8K"

    def test_construct_prompt_no_modifications(self):
        adapter_spec = TextToImageAdapterSpec(model="together/StableDiffusion", method=ADAPT_GENERATION, medium="photo")
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)
        eval_instance = Instance(Input(text="a blue dog"), references=[])
        prompt: Prompt = adapter.construct_prompt([], eval_instance, include_output=False, reference_index=None)

        assert adapter.window_service.fits_within_context_window(prompt.text)
        assert prompt.text == "a photo of a blue dog"
