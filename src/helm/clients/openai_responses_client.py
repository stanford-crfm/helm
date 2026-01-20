from typing import Any, Dict, List, Optional, Union

from helm.clients.openai_client import OpenAIClientUtils
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hwarn
from helm.common.media_object import TEXT_TYPE
from helm.common.request import (
    ErrorFlags,
    Thinking,
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
)
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.clients.client import (
    CachingClient,
    generate_uid_for_multimodal_prompt,
)

try:
    import openai
    from openai import OpenAI
    from openai.types.responses.response import Response
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class OpenAIResponseClient(CachingClient):
    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        openai_model_name: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.client = OpenAI(
            api_key=api_key,
            organization=org_id,
            base_url=base_url,
        )
        self.reasoning_effort = reasoning_effort
        self.openai_model_name = openai_model_name

    def _get_cache_key(self, raw_request: Dict, request: Request):
        cache_key = CachingClient.make_cache_key(raw_request, request)
        if request.multimodal_prompt:
            prompt_key: str = generate_uid_for_multimodal_prompt(request.multimodal_prompt)
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}
        return cache_key

    def _make_raw_request(self, request: Request) -> dict[str, Any]:
        input: Union[str, List[Dict[str, Any]]]

        if (
            (request.prompt and request.messages)
            or (request.prompt and request.multimodal_prompt)
            or (request.messages and request.multimodal_prompt)
        ):
            raise ValueError(
                f"More than one of `prompt`, `messages` and `multimodal_prompt` was set in request: {request}"
            )

        if request.messages is not None:
            # Checks that all messages have a role and some content
            for message in request.messages:
                if not message.get("role") or not message.get("content"):
                    raise ValueError("All messages must have a role and content")
            # Checks that the last role is "user"
            if request.messages[-1]["role"] != "user":
                raise ValueError("Last message must have role 'user'")
            if request.prompt != "":
                hwarn("Since message is set, prompt will be ignored")
            input = request.messages
        elif request.multimodal_prompt is not None:
            content = []
            request.validate()
            for media_object in request.multimodal_prompt.media_objects:
                if media_object.is_type("image") and media_object.location:
                    from helm.common.images_utils import encode_base64

                    base64_image: str = encode_base64(media_object.location)
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
                elif media_object.is_type(TEXT_TYPE):
                    assert media_object.text is not None
                    content.append({"type": "input_text", "text": media_object.text})
                else:
                    raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
            input = [{"role": "user", "content": content}]
        else:
            input = request.prompt

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "input": input,
            "top_p": request.top_p,
            # API errors if max_output_tokens is less than 16
            # (Error you get: "Invalid 'max_output_tokens': integer below minimum value.
            #    Expected a value >= 16, but got 5 instead.")
            "max_output_tokens": max(16, request.max_tokens),
            "temperature": request.temperature,
            # Don't store responses for later retrieval
            "store": False,
        }
        if self.reasoning_effort:
            raw_request["reasoning"] = {"effort": self.reasoning_effort}
        # If o-series model, get reasoning summaries
        # Plus other changes
        model_engine: str = request.model_engine
        if OpenAIClientUtils.is_reasoning_model(model_engine):
            if "reasoning" not in raw_request:
                raw_request["reasoning"] = {}
            raw_request["reasoning"]["summary"] = "detailed"
            # Avoid error:
            # "Error code: 400 - {'error': {'message': "Unsupported parameter: 'temperature' is
            # not supported with this model.", 'type': 'invalid_request_error', 'param': 'temperature',
            # 'code': 'unsupported_parameter'}}"
            raw_request.pop("temperature", None)

            # The following parameters also happen to be unsupported by the o-series (code unsupported_parameter)
            raw_request.pop("top_p", None)

        return raw_request

    def _get_model_for_request(self, request: Request) -> str:
        return self.openai_model_name or request.model_engine

    def make_request(self, request: Request) -> RequestResult:
        # Content can either be text or a list of multimodal content made up of text and images:
        # https://platform.openai.com/docs/api-reference/responses/create
        raw_request = self._make_raw_request(request)

        # The responses API does not support a "num_completions" parameter,
        # so we need to handle it ourselves with a simple loop
        completions: list[GeneratedOutput] = []
        for _ in range(request.num_completions):

            def do_it() -> Dict[str, Any]:
                raw_response = self.client.responses.create(**raw_request).model_dump(mode="json")
                assert not raw_response.get("error", None), f"Error in response: {raw_response}"
                return raw_response

            try:
                cache_key = self._get_cache_key(raw_request, request)
                helm_raw_response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except openai.OpenAIError as e:
                return OpenAIClientUtils.handle_openai_error(e, request)

            request_time = helm_raw_response["request_time"]
            del helm_raw_response["request_time"]
            request_datetime = helm_raw_response["request_datetime"]
            del helm_raw_response["request_datetime"]
            response = Response.model_validate(helm_raw_response)

            reasoning_output_parts: List[str] = []
            text_output_parts: List[str] = []

            if request.echo_prompt:
                text_output_parts.append(request.prompt)
            for output in response.output:

                if output.type == "reasoning":
                    for summary in output.summary:
                        reasoning_output_parts.append(summary.text)
                    if output.content:
                        for reasoning_content in output.content:
                            reasoning_output_parts.append(reasoning_content.text)
                elif output.type == "message":
                    for content in output.content:
                        if content.type == "output_text":
                            text_output_parts.append(content.text)
                        elif content.type == "refusal":
                            return RequestResult(
                                success=False,
                                cached=False,
                                error=f"Received refusal from OpenAI API: {content.refusal}",
                                completions=[],
                                embedding=[],
                                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                            )
                else:
                    raise ValueError(f"Unknown output type {output.type}")

            thinking = Thinking(text="\n\n".join(reasoning_output_parts)) if reasoning_output_parts else None
            text_output = "\n\n".join(text_output_parts)

            completions.append(
                GeneratedOutput(
                    text=text_output,
                    logprob=0.0,
                    tokens=[],
                    thinking=thinking,
                )
            )

        return RequestResult(
            success=True,
            cached=cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
