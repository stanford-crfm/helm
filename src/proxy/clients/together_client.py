from typing import List, Dict

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, truncate_sequence


class TogetherClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Uses the same parameter names as the OpenAI API: https://beta.openai.com/docs/api-reference/completions
        return {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
        }

    def __init__(self, cache_path: str):
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = TogetherClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def do_it():
                raise RuntimeError(
                    f"The result has not been uploaded to the cache ({self.cache.cache_path}) "
                    f"for the following request: {cache_key}"
                )

            response, cached = self.cache.get(cache_key, do_it)
        except RuntimeError as e:
            error: str = f"TogetherClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[])

        # Expect the result to be structured the same way as a response from OpenAI API.
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

        batch_performance_metadata: Dict = response["request_time"]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=0,
            completions=completions,
            batch_size=batch_performance_metadata["batch_size"],
            batch_request_time=batch_performance_metadata["batch_time"],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
