# mypy: check_untyped_defs = False
import dataclasses
from typing import Any, Dict, List, Optional, Union


from helm.clients.openai_client import OpenAIClientUtils
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hwarn
from helm.common.media_object import TEXT_TYPE
from helm.common.request import (
    Thinking,
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
)
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.clients.client import (
    CachingClient,
    truncate_and_tokenize_response_text,
    generate_uid_for_multimodal_prompt,
)
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class OpenAIResponseClient(CachingClient):
    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        openai_model_name: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
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
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except openai.OpenAIError as e:
                return OpenAIClientUtils.handle_openai_error(e, request)

            # We can only return one completition really,
            # but we get an array of messages back, so we need to contact them
            reasoning_output = ""
            text_output = ""

            if request.echo_prompt:
                text_output += request.prompt
            for output in response["output"]:
                output_type = output[
                    "type"
                ]  # one of "message" or "reasoning" from API observation, but can also include tool calls

                if output_type == "reasoning":
                    reasoning_output += "\n\n".join([raw_output["text"] for raw_output in output["summary"]])
                elif output_type == "message":
                    text_output += "\n\n".join([raw_output["text"] for raw_output in output["content"]])
                # (Other output types are ignored)

            completion = truncate_and_tokenize_response_text(
                text_output,
                request,
                self.tokenizer,
                self.tokenizer_name,
                original_finish_reason="",
            )
            if reasoning_output:
                completion = dataclasses.replace(completion, thinking=Thinking(text=reasoning_output))
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )
