import os
import json
import urllib.request
import ssl
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    List,
    cast,
    Optional
)

from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer

from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,

    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)

from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)

from .client import CachingClient, truncate_sequence
from openai import OpenAI
from helm.clients.openai_client import OpenAIClient


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    ssl._create_default_https_context = ssl._create_unverified_context
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


class AzureLlama3Client(CachingClient):
    """Implements a client for Azure hosted models like llama-70B"""
    def __init__(
            self,
            tokenizer: Tokenizer,
            tokenizer_name: str,
            cache_config: CacheConfig,
            timeout: int = 3000,
            do_cache: bool = False,
            api_key: str = None,
            endpoint: str = None,
    ):
        super().__init__(cache_config=cache_config)
        allowSelfSignedHttps(True)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.timeout = timeout
        self.do_cache = do_cache
        self.api_key = api_key
        self.api_endpoint = endpoint

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        # This needs to match whatever we define in pedantic
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        messages = [{"role": "user", "content": request.prompt}]

        raw_request = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
        }

        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + self.api_key)}

        body = str.encode(json.dumps(raw_request))
        req = urllib.request.Request(self.api_endpoint, body, headers)

        try:
            def do_it() -> Dict[str, Any]:
                response = urllib.request.urlopen(req)
                response_data = json.loads(response.read())
                return response_data

            if self.do_cache:
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                response, cached = do_it(), False

                completions: List[GeneratedOutput] = []
                for raw_completion in response["choices"]:
                    # The OpenAI chat completion API doesn't support echo.
                    # If `echo_prompt` is true, combine the prompt and completion.
                    raw_completion_content = raw_completion["message"]["content"]
                    text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
                    # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.

                    tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                        TokenizationRequest(text, tokenizer=self.tokenizer_name)
                    )

                    # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
                    tokens: List[Token] = [
                        Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
                    ]
                    completion = GeneratedOutput(
                        text=text,
                        logprob=0,
                        tokens=tokens,
                        finish_reason={"reason": raw_completion["finish_reason"]},
                    )
                    completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences

                    return RequestResult(
                        success=True,
                        cached=cached,
                        completions=completions,
                        embedding=[],
                    )
        except urllib.error.HTTPError as error:
            error_str: str = f"Request error: {error}"
            return RequestResult(success=False, cached=False, error=error_str, completions=[], embedding=[])
