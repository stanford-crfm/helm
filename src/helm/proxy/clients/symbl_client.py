from typing import List, Dict
from dataclasses import replace, asdict
import requests
from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT


class SymblClient(Client):

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.api_base: str = "http://127.0.0.1:9000"

    @staticmethod
    def tokenize_by_space(text: str) -> List[str]:
        """Simply tokenizes by a single white space."""
        return text.split(" ")

    def make_request(self, request: Request) -> RequestResult:
        # print(request)
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "prompt": request.prompt,
            "generation_parameters": {
                "max_new_tokens": 512,
                "penalty_alpha":request.penalty_alpha,
                "top_k":request.top_k
            }
        }

        def do_it():
            response = requests.post(self.api_base + "/v1/generate", json=raw_request).json()
            if "output" not in response:
                raise Exception("Invalid response from Symbl API: " + str(response))

            return response

        try:
            cache_key = Client.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        tokenization_result: TokenizationRequestResult = self.tokenize(
            TokenizationRequest(response["output"], tokenizer="huggingface/gpt2")
        )

        tokens: List[Token] = [
            Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
        ]

        completions = [
            Sequence(
                text=response["output"],
                logprob=0,
                tokens=tokens,
            )
        ]
        print(completions)
        return RequestResult(
            success=True,
            cached=cached,
            request_time=0,
            request_datetime=response.get("request_datetime"),
            #request_datetime = "request_datetime",
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        response = requests.post(self.api_base + "/v1/tokenize", json={"text": request.text}).json()
        return TokenizationRequestResult(
            success=True, cached=False, tokens=[TokenizationToken(text) for text in response["tokens"]], text=request.text
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
      
        try:
            def do_it():
                response = requests.post(self.api_base + "/v1/decode", json={"tokens":request.tokens}).json()
                return response
            cache_key = asdict(request) 
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"Symbl decode error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")
        print(result)
        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time="request_datetime"
        )


