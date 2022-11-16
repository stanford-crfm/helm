from typing import List, Optional

import openai as gooseai

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES


class GooseAIClient(Client):
    """
    GooseAI API Client
    - How to use the API: https://goose.ai/docs/api
    - Supported models: https://goose.ai/docs/models
    """

    def __init__(self, api_key: str, cache_config: CacheConfig, org_id: Optional[str] = None):
        self.org_id: Optional[str] = org_id
        self.api_key: str = api_key
        self.api_base: str = "https://api.goose.ai/v1"

        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        """
        Request parameters for GooseAI API documented here: https://goose.ai/docs/api/completions
        The only OpenAI API parameter not supported is `best_of`.
        """
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

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
                # Following https://beta.openai.com/docs/api-reference/authentication
                # `organization` can be set to None.
                gooseai.organization = self.org_id
                gooseai.api_key = self.api_key
                gooseai.api_base = self.api_base
                gooseai.api_resources.completion.Completion.__bases__ = ORIGINAL_COMPLETION_ATTRIBUTES
                return gooseai.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except gooseai.error.OpenAIError as e:
            error: str = f"OpenAI (GooseAI API) error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for text, logprob, top_logprobs in zip(
                raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
            ):
                tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                sequence_logprob += logprob or 0

            completion = Sequence(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
