from typing import List

import openai as gooseai

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken
from .client import Client, wrap_request_time
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES
from .tokenizer.tokenizer import Tokenizer
from .tokenizer.tokenizer_factory import TokenizerFactory


class GooseAIClient(Client):
    """
    GooseAI API Client
    - How to use the API: https://goose.ai/docs/api
    - Supported models: : https://goose.ai/docs/models
    """

    def __init__(self, api_key: str, cache_path: str):
        self.api_key: str = api_key
        self.api_base: str = "https://api.goose.ai/v1"

        self.cache = Cache(cache_path)
        self.tokenizer: Tokenizer = TokenizerFactory.get_tokenizer("gooseai")

    def make_request(self, request: Request) -> RequestResult:
        """
        Request parameters for GooseAI API documented here: https://goose.ai/docs/api/completions
        The only OpenAI API parameter not supported is `best_of`.
        """
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo": request.echo_prompt,
        }

        try:

            def do_it():
                gooseai.api_key = self.api_key
                gooseai.api_base = self.api_base
                gooseai.api_resources.completion.Completion.__bases__ = ORIGINAL_COMPLETION_ATTRIBUTES
                return gooseai.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except gooseai.error.OpenAIError as e:
            error: str = f"OpenAI (GooseAI API) error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[])

        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for text, logprob, top_logprobs in zip(
                raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
            ):
                tokens.append(Token(text=text, logprob=logprob, top_logprobs=dict(top_logprobs)))
                sequence_logprob += logprob or 0
            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completions.append(completion)

        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes the text using the GPT-2 tokenizer."""
        return TokenizationRequestResult(
            cached=False, tokens=[TokenizationToken(raw_text) for raw_text in self.tokenizer.tokenize(request.text)]
        )
