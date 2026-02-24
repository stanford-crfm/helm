import dataclasses
from typing import Any, Dict, List, Optional, TypedDict

from huggingface_hub import ChatCompletionOutput, InferenceClient

from helm.common.cache import CacheConfig
from helm.common.request import (
    Thinking,
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
)
from helm.clients.client import CachingClient


class HuggingFaceInferenceProvidersChatCompletionRequest(TypedDict):
    model: str
    messages: List[Dict]
    frequency_penalty: Optional[float]
    logprobs: Optional[bool]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[float]
    # TODO: Support JSON Schema response format
    # response_format: Optional[ChatCompletionInputGrammarType] = None
    stop: Optional[List[str]]
    temperature: Optional[float]
    top_p: Optional[float]


class HuggingFaceInferenceProvidersClient(CachingClient):

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        huggingface_model_name: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self._client = InferenceClient(api_key=api_key, base_url=base_url)
        self._huggingface_model_name = huggingface_model_name

    def _make_raw_request(self, request: Request) -> HuggingFaceInferenceProvidersChatCompletionRequest:
        input_messages: List[Dict[str, Any]]

        if request.multimodal_prompt:
            raise ValueError("`multimodal_prompt` is not supported by `HuggingFaceInferenceProvidersClient`")

        if request.prompt and request.messages:
            raise ValueError("More than one of `prompt` and `messages` was set in request")

        if request.messages is not None:
            # Checks that all messages have a role and some content
            for message in request.messages:
                if not message.get("role") or not message.get("content"):
                    raise ValueError("All messages must have a role and content")
            # Checks that the last role is "user"
            if request.messages[-1]["role"] != "user":
                raise ValueError("Last message must have role 'user'")
            input_messages = request.messages
        else:
            input_messages = [{"role": "user", "content": request.prompt}]

        return {
            "model": self._huggingface_model_name or request.model,
            "messages": input_messages,
            "frequency_penalty": request.frequency_penalty,
            "logprobs": False,
            "max_tokens": request.max_tokens,
            "n": request.num_completions,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop_sequences,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    def make_request(self, request: Request) -> RequestResult:
        if request.echo_prompt:
            raise NotImplementedError("`echo_prompt` is not supported")
        if request.embedding:
            raise NotImplementedError("`embedding` is not supported")

        raw_request = self._make_raw_request(request)

        def do_it() -> Dict[str, Any]:
            hf_raw_response = self._client.chat_completion(**raw_request)
            assert isinstance(hf_raw_response, ChatCompletionOutput)
            return dataclasses.asdict(hf_raw_response)

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        request_time = response["request_time"]
        del response["request_time"]
        request_datetime = response["request_datetime"]
        del response["request_datetime"]

        completions: list[GeneratedOutput] = []
        for choice in response["choices"]:
            thinking = Thinking(text=choice["message"]["reasoning"]) if choice["message"]["reasoning"] else None
            output_text = choice["message"]["content"]
            if output_text is None:
                raise ValueError("Response content was `None`, possibly due to content blocking")
            completion = GeneratedOutput(
                text=output_text,
                logprob=0.0,
                tokens=[],
                finish_reason={"reason": choice["finish_reason"]},
                thinking=thinking,
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
