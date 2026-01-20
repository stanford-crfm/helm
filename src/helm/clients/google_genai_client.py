from typing import Any, Dict, Optional, List

from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.media_object import AUDIO_TYPE, IMAGE_TYPE, TEXT_TYPE, VIDEO_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, ErrorFlags, Thinking
from helm.proxy.retry import NonRetriableException

try:
    from google.genai import Client
    from google.genai.types import (
        Candidate,
        Content,
        FinishReason,
        GenerateContentConfig,
        GenerateContentResponse,
        Part,
        ThinkingConfig,
    )
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google"])


class GoogleGenAIContentBlockedError(Exception):
    pass


class GoogleGenAIClient(CachingClient):
    """Client for Vertex AI models"""

    _CONTENT_BLOCKED_FINISH_REASONS: List[FinishReason] = [
        FinishReason.SAFETY,
        FinishReason.RECITATION,
        FinishReason.BLOCKLIST,
        FinishReason.PROHIBITED_CONTENT,
        FinishReason.SPII,
    ]

    _FINISH_REASON_MAPPING = {
        FinishReason.STOP: "stop",
        FinishReason.MAX_TOKENS: "length",
    }

    _ROLE_MAPPING: Dict[str, str] = {"user": "user", "assistant": "model"}

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        thinking_config: Optional[Dict[str, Any]] = None,
        genai_model: Optional[str] = None,
        genai_use_vertexai: Optional[bool] = None,
    ) -> None:
        super().__init__(cache_config=cache_config)
        self.project_id = project_id
        self.location = location
        self.genai_model = genai_model
        self.thinking_config = ThinkingConfig.model_validate(thinking_config) if thinking_config else None
        if genai_use_vertexai is True:
            hlog("GoogleGenAIClient using Vertex AI API with configured project ID and location")
            self.client = Client(vertexai=True, project=project_id, location=location)
        elif genai_use_vertexai is False:
            hlog("GoogleGenAIClient using Gemini API with configured API key")
            self.client = Client(api_key=api_key)
        elif project_id and location:
            hlog("GoogleGenAIClient using Vertex AI API with configured project ID and location")
            self.client = Client(vertexai=True, project=project_id, location=location)
        elif api_key:
            hlog("GoogleGenAIClient using Gemini API with configured API key")
            self.client = Client(api_key=api_key)
        else:
            hlog("GoogleGenAIClient using default google.genai.Client")
            self.client = Client()

    def _convert_request_to_contents(self, request: Request) -> List[Content]:
        contents: List[Content] = []
        if request.messages is not None:
            if request.multimodal_prompt or request.prompt:
                raise NonRetriableException(
                    "Only one of Request.prompt, Request.messages or Request.multimodal_prompt may be set"
                )
            for message in request.messages:
                if message["role"] == "system":
                    continue
                contents.append(
                    Content(
                        role=self._ROLE_MAPPING[message["role"]],
                        parts=[Part.from_text(text=message["content"])],
                    )
                )
        elif request.multimodal_prompt is not None:
            if request.messages or request.prompt:
                raise NonRetriableException(
                    "Only one of Request.prompt, Request.messages or Request.multimodal_prompt may be set"
                )
            for media_object in request.multimodal_prompt.media_objects:
                if (
                    media_object.is_type(IMAGE_TYPE)
                    or media_object.is_type(AUDIO_TYPE)
                    or media_object.is_type(VIDEO_TYPE)
                ):
                    if not media_object.location:
                        raise NonRetriableException(f"MediaObject.location must be set in MediaObject: {media_object}")
                    with open(media_object.location, "rb") as fp:
                        media_object_bytes = fp.read()
                    contents.append(
                        Content(
                            role="user",
                            parts=[Part.from_bytes(data=media_object_bytes, mime_type=media_object.content_type)],
                        )
                    )
                elif media_object.is_type(TEXT_TYPE):
                    if not media_object.text:
                        raise NonRetriableException(
                            f"MediaObject.text must be set in for MediaObject of type text: {media_object}"
                        )
                    contents.append(Content(role="user", parts=[Part.from_text(text=media_object.text)]))
                else:
                    raise Exception(f"MediaObject {media_object} has unknown type {media_object.type}")
        else:
            contents.append(Content(role=self._ROLE_MAPPING.get("user"), parts=[Part.from_text(text=request.prompt)]))
        return contents

    def _convert_request_to_generate_content_config(self, request: Request) -> GenerateContentConfig:
        # JSON Schema
        response_mime_type: Optional[str] = None
        response_json_schema: Optional[Dict[str, Any]] = None
        if request.response_format and request.response_format.json_schema:
            response_mime_type = "application/json"
            response_json_schema = request.response_format.json_schema

        system_instruction: Optional[str] = None
        if request.messages is not None:
            if request.messages[0]["role"] == "system":
                system_instruction = request.messages[0]["content"]

        seed: Optional[int] = None
        if request.random is not None:
            try:
                seed = int(request.random)
            except ValueError as e:
                raise NonRetriableException("Request.random must be an int for GoogleGenAIClient") from e
        return GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k_per_token,
            candidate_count=request.num_completions,
            max_output_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=seed,
            response_mime_type=response_mime_type,
            response_json_schema=response_json_schema,
            thinking_config=self.thinking_config,
            # Cannot request logprobs becuse it results in an error:
            # google.genai.errors.ServerError: 500 INTERNAL. {'error': {'code': 500, 'message': 'Missing Logprobs results.', 'status': 'INTERNAL'}}
            # response_logprobs=True,
            # logprobs=request.top_k_per_token,
        )

    def _convert_generate_content_response_to_generated_outputs(
        self, response: GenerateContentResponse
    ) -> List[GeneratedOutput]:
        # Content blocking can show up in many ways, so this defensively handles a few of these ways
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise GoogleGenAIContentBlockedError(f"Prompt blocked with reason: {response.prompt_feedback.block_reason}")
        if not response.candidates:
            raise GoogleGenAIContentBlockedError(f"No candidates in response: {response}")

        generated_outputs: List[GeneratedOutput] = []
        for candidate in response.candidates:
            assert isinstance(candidate, Candidate)
            assert candidate.content
            assert candidate.content.role == "model" or candidate.content.role is None

            # Content blocking can show up in many ways, so this defensively handles a few of these ways
            if candidate.finish_reason in self._CONTENT_BLOCKED_FINISH_REASONS:
                raise GoogleGenAIContentBlockedError(f"Content blocked with reason: {candidate.finish_reason}")
            if not candidate.content:
                raise GoogleGenAIContentBlockedError(f"No content in candidate: {candidate}")
            if not candidate.content.parts:
                if candidate.finish_reason != FinishReason.MAX_TOKENS:
                    raise GoogleGenAIContentBlockedError(f"No content parts in candidate: {candidate}")

            thinking_text_parts: List[str] = []
            output_text_parts: List[str] = []
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text is not None:
                        if part.thought:
                            thinking_text_parts.append(part.text)
                        else:
                            output_text_parts.append(part.text)
            thinking_text = "".join(thinking_text_parts)
            output_text = "".join(output_text_parts)

            finish_reason = (
                {"reason": self._FINISH_REASON_MAPPING.get(candidate.finish_reason, candidate.finish_reason)}
                if candidate.finish_reason
                else None
            )

            generated_outputs.append(
                GeneratedOutput(
                    text=output_text,
                    logprob=0.0,  # API doesn't support logprobs or tokens
                    tokens=[],  # API doesn't support logprobs or tokens
                    finish_reason=finish_reason,
                    thinking=Thinking(text=thinking_text) if thinking_text else None,
                )
            )
        return generated_outputs

    def _get_model_name_for_request(self, request: Request) -> str:
        if self.genai_model is not None:
            return self.genai_model
        return request.model_engine

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        if request.echo_prompt:
            raise NotImplementedError("GoogleGenAIClient does not support Request.echo_prompt")

        model_name = self._get_model_name_for_request(request)
        contents = self._convert_request_to_contents(request)
        generate_content_config = self._convert_request_to_generate_content_config(request)

        def do_it() -> Dict[str, Any]:
            # NOTE: The type signature of `config` in `generate_content()` is wrong.
            # It should be `Sequence[Content | ...]` but instead it is `list[Content | ...]`
            # which is a problem because `list`` is type invariant.
            response: GenerateContentResponse = self.client.models.generate_content(
                model=model_name, contents=contents, config=generate_content_config  # type: ignore
            )
            return response.model_dump()

        cache_key_contents = (
            generate_uid_for_multimodal_prompt(request.multimodal_prompt)
            if request.multimodal_prompt
            else [c.model_dump() for c in contents]
        )
        cache_key = {
            "model": model_name,
            "contents": cache_key_contents,
            "config": generate_content_config.model_dump(),
        }

        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        request_time = response["request_time"]
        del response["request_time"]
        request_datetime = response["request_datetime"]
        del response["request_datetime"]
        generate_content_response = GenerateContentResponse.model_validate(response)

        try:
            completions = self._convert_generate_content_response_to_generated_outputs(generate_content_response)
        except GoogleGenAIContentBlockedError as e:
            hwarn(f"Content blocked: {str(e)}")
            return RequestResult(
                success=False,
                cached=False,
                error=f"Content blocked: {str(e)}",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            )

        return RequestResult(
            success=True,
            cached=cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
