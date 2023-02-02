from typing import List

from helm.common.tokenization_request import TokenizationToken
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .adapter_factory import AdapterFactory, ADAPT_LANGUAGE_MODELING
from .test_adapter import TestAdapter


class TestLanguageModelingAdapter(TestAdapter):
    def test_construct_language_modeling_prompt(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_LANGUAGE_MODELING,
            input_prefix="",
            model="openai/davinci",
            output_prefix="",
            max_tokens=0,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)

        # The tokens translate to: '�Excuse me�'
        conditioning_tokens: List[TokenizationToken] = [TokenizationToken(110), TokenizationToken(40127)]
        pred_tokens: List[TokenizationToken] = [TokenizationToken(1904), TokenizationToken(502), TokenizationToken(447)]
        prompt, num_conditioning_tokens = adapter.construct_language_modeling_prompt(
            conditioning_tokens=conditioning_tokens, pred_tokens=pred_tokens, max_req_len=5, text=""
        )

        # Ensure the prompt is correct
        assert prompt == "Excuse me"

        # Ensure the number of conditioning tokens is correct
        assert num_conditioning_tokens == 1

    def test_fits_tokens_within_context_window(self):
        adapter_spec = AdapterSpec(
            method=ADAPT_LANGUAGE_MODELING,
            input_prefix="",
            model="openai/curie",
            output_prefix="",
            max_tokens=0,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self.tokenizer_service)

        # The tokens translate to: '<|endoftext|>The the the the ... the the'
        # There are 1 `conditioning_token` and 2049 `pred_tokens`. Since the `max_request_length`
        # of GPT-3 is 2049, calling `fits_tokens_within_context_window` will remove the last `pred_token`
        conditioning_tokens: List[TokenizationToken] = [TokenizationToken(50256)]
        pred_tokens: List[TokenizationToken] = [TokenizationToken(464)] + [TokenizationToken(262)] * 2048
        prompt, pred_tokens = adapter.fits_tokens_within_context_window(
            conditioning_tokens, pred_tokens, adapter.window_service.max_request_length
        )

        # Ensure the prompt is correct
        assert prompt == "<|endoftext|>The" + " the" * 2047

        # Ensure the pred_tokens are correct
        assert pred_tokens == [TokenizationToken(464)] + [TokenizationToken(262)] * 2047
