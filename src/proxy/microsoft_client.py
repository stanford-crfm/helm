from typing import List, Optional

from filelock import FileLock
from openai.api_resources.abstract import engine_api_resource
import openai as turing

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES


class MicrosoftClient(Client):
    _CLIENT_LOCK = "microsoft_client.lock"

    """
    Client for the Microsoft's Megatron-Turing NLG models (https://arxiv.org/abs/2201.11990).

    According to the internal documentation: https://github.com/microsoft/turing-academic-TNLG,
    "the model will generate roughly 3 tokens per second. The response will be returned once
    all tokens have been generated."
    """

    def __init__(self, api_key: str, cache_path: str, org_id: Optional[str] = None):
        # Adapted from their documentation: https://github.com/microsoft/turing-academic-TNLG
        class EngineAPIResource(engine_api_resource.EngineAPIResource):
            @classmethod
            def class_url(
                cls, engine: Optional[str] = None, api_type: Optional[str] = None, api_version: Optional[str] = None
            ) -> str:
                return f"/{engine}/inference"

        self.org_id: Optional[str] = org_id
        self.api_key: str = api_key
        self.api_base: str = "https://turingnlg-turingnlg-mstap-v2.turingase.p.azurewebsites.net"
        self.completion_attributes = (EngineAPIResource,) + ORIGINAL_COMPLETION_ATTRIBUTES[1:]

        self.cache = Cache(cache_path)

        # The Microsoft Turing server only allows a single request at a time, so acquire a
        # process-safe lock before making a request.
        # https://github.com/microsoft/turing-academic-TNLG#rate-limitations
        #
        # Since the model will generate roughly three tokens per second and the max context window
        # is 2048 tokens, we expect the maximum time for a request to be fulfilled to be 700 seconds.
        self.lock = FileLock(MicrosoftClient._CLIENT_LOCK, timeout=700)

    def make_request(self, request: Request) -> RequestResult:
        """
        Make a request for the Microsoft MT-NLG models.

        They mimicked the OpenAI completions API, but not all the parameters are supported.

        Supported parameters:
            engine
            prompt
            temperature
            max_tokens
            best_of
            logprobs
            stop ("Only a single "stop" value (str) is currently supported.")
            top_p
            echo
            n (Not originally supported, but we simulate n by making multiple requests)

        Not supported parameters:
            presence_penalty
            frequency_penalty
        """
        # Only a single "stop" value (str) or None is currently supported.
        stop_sequence: Optional[str]
        if len(request.stop_sequences) == 0:
            stop_sequence = None
        elif len(request.stop_sequences) == 1:
            stop_sequence = request.stop_sequences[0]
        else:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": stop_sequence,
            "top_p": request.top_p,
            "echo": request.echo_prompt,
        }

        completions: List[Sequence] = []
        request_time = 0
        all_cached = True

        # API currently only supports 1 completion at a time, so we have to hit it multiple times.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    with self.lock:
                        # Following https://beta.openai.com/docs/api-reference/authentication
                        # `organization` can be set to None.
                        turing.organization = self.org_id
                        turing.api_key = self.api_key
                        turing.api_base = self.api_base
                        turing.api_resources.completion.Completion.__bases__ = self.completion_attributes
                        return turing.Completion.create(**raw_request)

                # We want to make `request.num_completions` fresh requests,
                # cache key should contain the completion_index.
                cache_key = Client.make_cache_key({"completion_index": completion_index, **raw_request}, request)
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except turing.error.OpenAIError as e:
                error: str = f"OpenAI (Turing API) error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[])

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
                completions.append(completion)

            request_time += response["request_time"]
            all_cached = all_cached and cached

        return RequestResult(success=True, cached=all_cached, request_time=request_time, completions=completions)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
