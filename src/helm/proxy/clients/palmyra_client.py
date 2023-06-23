from dataclasses import asdict
import json
import os
import requests
from typing import Any, Dict, List, Optional
from tempfile import TemporaryDirectory
from threading import Lock

from sentencepiece import SentencePieceProcessor

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult, Sequence, Token, ErrorFlags
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from helm.common.general import ensure_file_downloaded
from .client import Client, wrap_request_time, truncate_sequence


_TOKENIZER_MODEL_URL = "https://huggingface.co/spaces/Writer/token-counter/resolve/92389981e4430f83383110728d954bda0f89fb30/tokenizer.model"  # noqa


# SentencePieceProcessor is lazily initialized
_sentence_piece_processor: Optional[SentencePieceProcessor] = None
_sentence_piece_processor_directory: Optional[TemporaryDirectory] = None
_sentence_piece_processor_lock = Lock()


def _initialize_sentence_piece_processor() -> None:
    global _sentence_piece_processor
    global _sentence_piece_processor_directory
    global _sentence_piece_processor_lock
    with _sentence_piece_processor_lock:
        if _sentence_piece_processor:
            return
        _sentence_piece_processor_directory = TemporaryDirectory()
        sentence_piece_model_path = os.path.join(_sentence_piece_processor_directory.name, "palmrya_tokenizer.model")
        ensure_file_downloaded(_TOKENIZER_MODEL_URL, sentence_piece_model_path)
        _sentence_piece_processor = SentencePieceProcessor(sentence_piece_model_path)


class PalmyraClient(Client):
    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key: str = api_key
        self.cache = Cache(cache_config)

    def __del__(self):
        with self._sentence_piece_processor_lock:
            self._sentence_piece_processor = None
            if self._sentence_piece_processor_directory:
                self._sentence_piece_processor_directory.cleanup()
        super.__del__()

    def _send_request(self, model_name: str, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.request(
            method="POST",
            url=f"https://enterprise-api.writer.com/llm/organization/3002/model/{model_name}/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(raw_request),
        )
        result = json.loads(response.text)
        if "error" in result:
            raise ValueError(f"Request failed with error: {result['error']}")
        return result

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        raw_request = {
            "prompt": request.prompt,
            "maxTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": request.top_p,
            "bestOf": request.top_k_per_token,
            "stop": request.stop_sequences,
            # random_seed have been disabled for now.
            # It is here to ensure that Writer does not cache the request when we
            # want several completions with the same prompt. Right now it seems
            # to have no effect so we are disabling it.
            # TODO(#1515): re-enable it when it works.
            # "random_seed": request.random,
        }

        if request.random is not None or request.num_completions > 1:
            hlog(
                "WARNING: Writer does not support random_seed or num_completions. "
                "This request will be sent to Writer multiple times."
            )

        completions: List[Sequence] = []
        model_name: str = request.model_engine

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:
                # This is disabled for now. See above TODO(#1515).
                # HACKY: Use the random seed to get different results for each completion.
                # raw_request["random_seed"] = (
                #     f"completion_index={completion_index}"
                #     if request.random is None
                #     else request.random + f":completion_index={completion_index}"
                # )

                def do_it():
                    # Add an argument timeout to raw_request to avoid waiting getting timeout of 60s
                    # which happens for long prompts.
                    request_with_timeout = {"timeout": 300, **raw_request}
                    result = self._send_request(model_name, request_with_timeout)
                    return result

                # We need to include the engine's name to differentiate among requests made for different model
                # engines since the engine name is not included in the request itself.
                # In addition, we want to make `request.num_completions` fresh
                # requests, cache key should contain the completion_index.
                # Echoing the original prompt is not officially supported by Writer. We instead prepend the
                # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
                cache_key = Client.make_cache_key(
                    {
                        "engine": request.model_engine,
                        "completion_index": completion_index,
                        **raw_request,
                    },
                    request,
                )

                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"PalmyraClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            if "choices" not in response:
                if "errors" in response and response["errors"][0]["key"] == "fail.content.moderation.failed":
                    return RequestResult(
                        success=False,
                        cached=False,
                        error=response["errors"][0]["description"],
                        completions=[],
                        embedding=[],
                        error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                        request_time=response["request_time"],
                        request_datetime=response["request_datetime"],
                    )
                else:
                    raise ValueError(f"Invalid response: {response}")

            response_text: str = response["choices"][0]["text"]

            # The Writer API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            # The Writer API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenize(
                # Writer uses their own huggingface tokenizer
                TokenizationRequest(text, tokenizer="writer/palmyra-tokenizer")
            )

            # Log probs are not currently not supported by the Writer, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
            ]

            completion = Sequence(text=response_text, logprob=0, tokens=tokens)
            sequence = truncate_sequence(completion, request, print_warning=True)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                _initialize_sentence_piece_processor()
                if request.encode:
                    tokens = _sentence_piece_processor.EncodeAsIds(request.text)
                else:
                    tokens = [
                        # TODO: Replace this with a helper function after #1549 is merged.
                        piece.replace("▁", " ")
                        for piece in _sentence_piece_processor.EncodeAsPieces(request.text)
                    ]
                if request.truncation:
                    tokens = tokens[: request.max_length]
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"PalmyraClient tokenize error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        cache_key = asdict(request)

        try:

            def do_it():
                _initialize_sentence_piece_processor()
                return {"text": _sentence_piece_processor.DecodeIds(request.tokens)}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"PalmyraClient decode error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
