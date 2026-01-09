# mypy: check_untyped_defs = False
from typing import Any, Dict, List, Optional, TypedDict

from helm.common.cache import CacheConfig
from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
)
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.clients.client import CachingClient

try:
    import litellm
    from litellm.types.utils import Choices, ModelResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["litellm"])


class LiteLLMCompletionRequest(TypedDict):
    model: str
    messages: List
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stop: Optional[List[str]]
    max_tokens: Optional[int]
    logprobs: Optional[bool]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]


class LiteLLMCompletionClient(CachingClient):

    def __init__(
        self,
        cache_config: CacheConfig,
        litellm_model: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self._litellm_model = litellm_model

    def _make_raw_request(self, request: Request) -> LiteLLMCompletionRequest:
        input_messages: List[Dict[str, Any]]

        if request.multimodal_prompt:
            raise ValueError("`multimodal_prompt` is not supported by `LiteLLMClient`")

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
            "model": self._get_model_for_request(request),
            "messages": input_messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences,
            "max_tokens": request.max_tokens,
            "logprobs": bool(request.top_k_per_token),
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

    def _get_model_for_request(self, request: Request) -> str:
        return self._litellm_model or request.model_engine

    def make_request(self, request: Request) -> RequestResult:
        if request.echo_prompt:
            raise NotImplementedError("`echo_prompt` is not supported")
        if request.embedding:
            raise NotImplementedError("`embedding` is not supported")

        raw_request = self._make_raw_request(request)

        # The responses API does not support a "num_completions" parameter,
        # so we need to handle it ourselves with a simple loop
        completions: list[GeneratedOutput] = []
        for _ in range(request.num_completions):

            def do_it() -> Dict[str, Any]:
                litellm_raw_response = litellm.completion(**raw_request).model_dump(mode="json")
                assert not litellm_raw_response.get("error", None), f"Error in response: {litellm_raw_response}"
                return litellm_raw_response

            cache_key = CachingClient.make_cache_key(raw_request, request)
            helm_raw_response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            request_time = helm_raw_response["request_time"]
            del helm_raw_response["request_time"]
            request_datetime = helm_raw_response["request_datetime"]
            del helm_raw_response["request_datetime"]
            response = ModelResponse.model_validate(helm_raw_response)
            for choice in response.choices:
                assert isinstance(response.choices, Choices)
                output_text = choice.message.content
                if output_text is None:
                    raise ValueError("Response content was `None`, possibly due to content blocking")
                completion = GeneratedOutput(
                    text=output_text, logprob=0.0, tokens=[], finish_reason={"reason": choice.finish_reason}
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
