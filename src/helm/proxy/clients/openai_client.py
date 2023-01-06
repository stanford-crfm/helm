from dataclasses import replace
from typing import Any, Dict, List, Optional

import openai

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, truncate_sequence, wrap_request_time
from .chat_gpt_client import ChatGPTClient

ORIGINAL_COMPLETION_ATTRIBUTES = openai.api_resources.completion.Completion.__bases__


class OpenAIClient(Client):
    END_OF_TEXT: str = "<|endoftext|>"

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        chat_gpt_client: Optional[ChatGPTClient] = None,
        org_id: Optional[str] = None,
    ):
        self.org_id: Optional[str] = org_id
        self.api_key: str = api_key
        self.api_base: str = "https://api.openai.com/v1"
        self.cache = Cache(cache_config)
        self.chat_gpt_client: Optional[ChatGPTClient] = chat_gpt_client

    def make_request(self, request: Request) -> RequestResult:
        if request.model_engine == "chat-gpt":
            assert self.chat_gpt_client is not None
            return self.chat_gpt_client.make_request(request)

        raw_request: Dict[str, Any]
        if request.embedding:
            raw_request = {
                "input": request.prompt,
                "engine": request.model_engine,
            }
        else:
            raw_request = {
                "engine": request.model_engine,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "n": request.num_completions,
                "max_tokens": request.max_tokens,
                "best_of": request.top_k_per_token,
                "logprobs": request.top_k_per_token,
                "stop": request.stop_sequences or None,  # API doesn't like empty list
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "echo": request.echo_prompt,
            }

            # OpenAI doesn't let you ask for more completions than the number of
            # per-token candidates.
            raw_request["best_of"] = max(raw_request["best_of"], raw_request["n"])
            raw_request["logprobs"] = max(raw_request["logprobs"], raw_request["n"])

        try:
            if request.embedding:

                def do_it():
                    openai.organization = self.org_id
                    openai.api_key = self.api_key
                    openai.api_base = self.api_base
                    return openai.Embedding.create(**raw_request)

            else:

                def do_it():
                    # Following https://beta.openai.com/docs/api-reference/authentication
                    # `organization` can be set to None.
                    openai.organization = self.org_id
                    openai.api_key = self.api_key
                    openai.api_base = self.api_base
                    openai.api_resources.completion.Completion.__bases__ = ORIGINAL_COMPLETION_ATTRIBUTES
                    return openai.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.error.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # If the user is requesting completions instead of an embedding, then `completions`
        # needs to be populated, and `embedding` should be an empty list and vice-versa.
        embedding: List[float] = []
        completions: List[Sequence] = []
        if request.embedding:
            # If the user is requesting an embedding instead of completion
            # then completions would be left as an empty list. The embedding needs to be set.
            embedding = response["data"][0]["embedding"]
        else:
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
                # OpenAI sends us back tokens past the end of text token,
                # so we need to manually truncate the list of tokens.
                # TODO: filed an issue with their support to check what the expected behavior here is.
                completion = truncate_sequence(
                    completion, replace(request, stop_sequences=request.stop_sequences + [OpenAIClient.END_OF_TEXT])
                )
                completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=embedding,
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
