from typing import List, Dict

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, truncate_sequence


class GoogleClient(Client):
    """
    Client for the Google models. There isn't an API for their language models.
    We receive and process completions offline.
    """

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        return {
            "best_of": request.top_k_per_token,
            "echo": request.echo_prompt,
            "logprobs": request.top_k_per_token,
            "max_tokens": request.max_tokens,
            "model": request.model_engine,
            "n": request.num_completions,
            "prompt": request.prompt,
            "request_type": "language-model-inference",
            "stop": request.stop_sequences or None,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = GoogleClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def fail():
                raise RuntimeError(
                    f"The result has not been uploaded to the cache for the following request: {cache_key}"
                )

            # If results are not cached for a given query, fail fast
            response, cached = self.cache.get(cache_key, fail)
        except RuntimeError as e:
            error: str = f"GoogleClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for text, logprob in zip(raw_data["tokens"], raw_data["token_logprobs"]):
                tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict()))
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
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
