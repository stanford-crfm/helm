from typing import List, Optional

from filelock import FileLock
from openai.api_resources.abstract import engine_api_resource
import openai as turing

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken
from .client import Client, wrap_request_time
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES
from .tokenizer.tokenizer import Tokenizer
from .tokenizer.tokenizer_factory import TokenizerFactory


class MicrosoftClient(Client):
    _CLIENT_LOCK = "microsoft_client.lock"

    """
    Client for the Microsoft's Megatron-Turing NLG models (https://arxiv.org/abs/2201.11990).

    According to the internal documentation: https://github.com/microsoft/turing-academic-TNLG,
    "the model will generate roughly 3 tokens per second. The response will be returned once
    all tokens have been generated."
    """

    def __init__(self, api_key: str, cache_path: str):
        # Adapted from their documentation: https://github.com/microsoft/turing-academic-TNLG
        class EngineAPIResource(engine_api_resource.EngineAPIResource):
            @classmethod
            def class_url(
                cls, engine: Optional[str] = None, api_type: Optional[str] = None, api_version: Optional[str] = None
            ) -> str:
                return f"/{engine}/inference"

        self.api_key: str = api_key
        self.api_base: str = "https://turingnlg-turingnlg-mstap-v2.turingase.p.azurewebsites.net"
        self.completion_attributes = (EngineAPIResource,) + ORIGINAL_COMPLETION_ATTRIBUTES[1:]

        self.cache = Cache(cache_path)
        self.tokenizer: Tokenizer = TokenizerFactory.get_tokenizer("microsoft")

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
            stop ("Only a single "stop" value (str) is currently supported.")
            top_p
            echo

        Not supported parameters:
            n (to get multiple completions)
            best_of
            logprobs
            presence_penalty
            frequency_penalty

        Log probabilities are also currently not supported.
        """

        def fix_text(text: str) -> str:
            return text.replace("Ġ", " ")

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
            "stop": stop_sequence,
            "top_p": request.top_p,
            "echo": request.echo_prompt,
        }

        try:

            def do_it():
                with self.lock:
                    turing.api_key = self.api_key
                    turing.api_base = self.api_base
                    turing.api_resources.completion.Completion.__bases__ = self.completion_attributes
                    return turing.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except turing.error.OpenAIError as e:
            error: str = f"OpenAI (Turing API) error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[])

        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            # TODO: handle logprobs when it's supported (currently always null). Current example response:
            # {
            #   "finish_reason": "stop",
            #   "index": 0,
            #   "logprobs": null,
            #   "text": "So I was takin' a walk the other day"
            # }
            # Since the log probs and tokens are not available to us just tokenize the completion using the tokenizer
            completion_text: str = raw_completion["text"]
            tokens: List[Token] = [
                Token(text=fix_text(text), logprob=0, top_logprobs={})
                for text in self.tokenizer.tokenize(completion_text)
            ]
            completion = Sequence(text=completion_text, logprob=0, tokens=tokens)
            completions.append(completion)
        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes the text using the GPT-2 tokenizer created in `MTNLGTokenizer`."""
        return TokenizationRequestResult(
            cached=False, tokens=[TokenizationToken(raw_text) for raw_text in self.tokenizer.tokenize(request.text)]
        )
