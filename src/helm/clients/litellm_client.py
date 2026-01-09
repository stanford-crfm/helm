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
    from litellm.types.utils import ModelResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["litellm"])


class LiteLLMCompletionRequest(TypedDict):
    model: str
    messages: List
    temperature: Optional[float]
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_tokens: Optional[int] = None,
    logprobs: Optional[bool] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,

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
            raise ValueError(
                f"`multimodal_prompt` is not supported by `LiteLLMClient`"
            )

        if request.prompt and request.messages:
            raise ValueError(
                f"More than one of `prompt` and `messages` was set in request"
            )

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

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": input_messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences,
            "max_tokens": request.max_tokens,
            "logprobs": request.top_k_per_token,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        return raw_request

    def _get_model_for_request(self, request: Request) -> str:
        return self._litellm_model or request.model_engine

    def make_request(self, request: Request) -> RequestResult:
        if request.echo_prompt:
            return "`echo_prompt` is not supported"
        if request.embedding:
            return "`embedding` is not supported"

        # Content can either be text or a list of multimodal content made up of text and images:
        # https://platform.openai.com/docs/api-reference/responses/create
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
                output_text = choice.message.content
                if output_text is None:
                    raise ValueError("Response content was `None`, possibly due to content blocking")
                completion = GeneratedOutput(text=output_text, logprob=0.0, tokens=[], finish_reason={"reason": choice.finish_reason})
                completions.append(completion)
            #     choice.logprobs
            #     choice.message
                

            #     # The OpenAI chat completion API doesn't support echo.
            #     # If `echo_prompt` is true, combine the prompt and completion.
            #     raw_completion_content = raw_completion["message"]["content"]
            #     if self.output_processor:
            #         raw_completion_content = self.output_processor(raw_completion_content)
            #     text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
            #     # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
            #     tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
            #         TokenizationRequest(text, tokenizer=self.tokenizer_name)
            #     )
            #     # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
            #     tokens: List[Token] = [
            #         Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
            #     ]
            #     # vLLM has a optional `reasoning_content` field in the message
            #     # that is not in the standard OpenAI API.
            #     # This field is also used by some model providers such as Grok.
            #     thinking = (
            #         Thinking(text=raw_completion["message"]["reasoning_content"])
            #         if "reasoning_content" in raw_completion["message"]
            #         else None
            #     )
            #     completion = GeneratedOutput(
            #         text=,
            #         logprob=0,  # OpenAI does not provide logprobs
            #         tokens=tokens,
            #         finish_reason={"reason": choice.finish_reason},
            #         thinking=thinking,
            #     )
            #     completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences

            

            # # We can only return one completition really,
            # # but we get an array of messages back, so we need to handle them

            # reasoning_output = ""
            # text_output = ""

            # if request.echo_prompt:
            #     text_output += request.prompt
            # for output in response.output:
            #     output_type = (
            #         output.type
            #     )  # one of "message" or "reasoning" from API observation, but can also include tool calls

            #     if output_type == "reasoning":
            #         reasoning_output += "\n\n".join([summary.text for summary in output.summary])
            #     elif output_type == "message":
            #         text_output += "\n\n".join([content.text for content in output.content])
            #     # (Other output types are ignored)

            # completion = GeneratedOutput(text=text_output, logprob=0.0, tokens=[], finish_reason={"reason": ""})
            # if reasoning_output:
            #     completion = dataclasses.replace(completion, thinking=Thinking(text=reasoning_output))
            # completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
