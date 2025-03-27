# mypy: check_untyped_defs = False
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast, Union, Callable

from helm.benchmark.model_metadata_registry import is_vlm
from helm.common import multimodal_request_utils
from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE, MultimediaObject
from helm.common.request import ErrorFlags, wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import get_class_by_name
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import Client, CachingClient, truncate_sequence, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class OpenAIClient(CachingClient):
    END_OF_TEXT: str = "<|endoftext|>"

    # Error OpenAI throws when the image in the prompt violates their content policy
    INAPPROPRIATE_IMAGE_ERROR: str = "Your input image may contain content that is not allowed by our safety system"
    INAPPROPRIATE_PROMPT_ERROR: str = "Invalid prompt: your prompt was flagged"
    INAPPROPRIATE_PROMPT_AZURE_ERROR: str = (
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
    )
    INAPPROPRIATE_PROMPT_MICROSOFT_ERROR: str = (
        "The response was filtered due to the prompt triggering Microsoft's content management policy."
    )

    # OpenAI server error
    OPENAI_SERVER_ERROR: str = (
        "The server had an error processing your request. Sorry about that! You can retry your request, "
        "or contact us through our help center at help.openai.com if you keep seeing this error."
    )

    # Set the finish reason to this if the prompt violates OpenAI's content policy
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = (
        "The prompt violates OpenAI's content policy. "
        "See https://labs.openai.com/policies/content-policy for more information."
    )

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
        output_processor: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.client = OpenAI(api_key=api_key, organization=org_id, base_url=base_url)
        self.reasoning_effort = reasoning_effort
        self.openai_model_name = openai_model_name
        self.output_processor: Optional[Callable[[str], str]] = (
            get_class_by_name(output_processor) if output_processor else None
        )

    def _get_model_for_request(self, request: Request) -> str:
        return self.openai_model_name or request.model_engine

    def _get_cache_key(self, raw_request: Dict, request: Request):
        cache_key = CachingClient.make_cache_key(raw_request, request)
        if request.multimodal_prompt:
            prompt_key: str = generate_uid_for_multimodal_prompt(request.multimodal_prompt)
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}

            if "messages" in cache_key:
                del cache_key["messages"]
        return cache_key

    def _make_embedding_request(self, request: Request) -> RequestResult:
        raw_request: Dict[str, Any]
        raw_request = {
            "input": request.prompt,
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
        }

        def do_it() -> Dict[str, Any]:
            return self.client.embeddings.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # If the user is requesting completions instead of an embedding, then `completions`
        # needs to be populated, and `embedding` should be an empty list and vice-versa.
        embedding: List[float] = []
        # If the user is requesting an embedding instead of completion
        # then completions would be left as an empty list. The embedding needs to be set.
        embedding = response["data"][0]["embedding"]

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[],
            embedding=embedding,
        )

    def _make_chat_request(self, request: Request) -> RequestResult:
        messages: Optional[List[Dict[str, Union[str, Any]]]] = request.messages
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
                hlog("WARNING: Since message is set, prompt will be ignored")
        else:
            # Convert prompt into a single message
            # For now, put the whole prompt in a single user message, and expect the response
            # to be returned in a single assistant message.
            # TODO: Support ChatML for creating multiple messages with different roles.
            # See: https://github.com/openai/openai-python/blob/main/chatml.md

            # Content can either be text or a list of multimodal content made up of text and images:
            # https://platform.openai.com/docs/guides/vision
            content: Union[str, List[Union[str, Any]]]
            if request.multimodal_prompt is not None:
                content = []
                request.validate()
                for media_object in request.multimodal_prompt.media_objects:
                    if media_object.is_type("image") and media_object.location:
                        from helm.common.images_utils import encode_base64

                        base64_image: str = encode_base64(media_object.location)
                        image_object: Dict[str, str] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                        content.append({"type": "image_url", "image_url": image_object})
                    elif media_object.is_type("audio") and media_object.location:
                        base64_audio: str = multimodal_request_utils.get_contents_as_base64(media_object.location)
                        format: str = media_object.content_type.split("/")[1]
                        if format == "mpeg":
                            # OpenAI expects "mp3" for mpeg audio
                            format = "mp3"

                        content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {"data": base64_audio, "format": format},
                            }
                        )
                    elif media_object.is_type(TEXT_TYPE):
                        content.append({"type": media_object.type, "text": media_object.text})
                    else:
                        raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

            else:
                content = request.prompt

            messages = [{"role": "user", "content": content}]

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            # Note: Chat models may require adding an extra token to max_tokens
            # for the internal special role token.
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        if request.response_format and request.response_format.json_schema:
            # Copy and modify JSON schema to conform to OpenAI's requirements
            json_schema = dict(request.response_format.json_schema)

            # additionalProperties: false must always be set in objects
            # https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
            if "additionalProperties" not in json_schema:
                json_schema["additionalProperties"] = False

            # All fields must be required
            # https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
            if "required" not in json_schema:
                json_schema["required"] = list(json_schema["properties"].keys())

            raw_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "description": "",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # Special handling for o1 models.
        # Refer to the "Reasoning models" documentation further discussion of o1 model limitations:
        # https://platform.openai.com/docs/guides/reasoning
        model_engine: str = request.model_engine
        if model_engine.startswith("o1") or model_engine.startswith("o3"):
            # Avoid error:
            # "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."  # noqa: E501
            # Note that openai>=1.45 is needed for this
            if raw_request["max_tokens"]:
                raw_request["max_completion_tokens"] = raw_request["max_tokens"]
                raw_request.pop("max_tokens")
            # Avoid error:
            # "Invalid type for 'stop': expected an unsupported value, but got null instead."
            if raw_request["stop"] is None:
                raw_request.pop("stop")

            # Avoid error:
            # "Error code: 400 - {'error': {'message': "Unsupported parameter: 'temperature' is
            # not supported with this model.", 'type': 'invalid_request_error', 'param': 'temperature',
            # 'code': 'unsupported_parameter'}}"
            raw_request.pop("temperature", None)

            if self.reasoning_effort:
                raw_request["reasoning_effort"] = "self.reasoning_effort"
        elif is_vlm(request.model):
            # Avoid error:
            # "Invalid type for 'stop': expected an unsupported value, but got null instead."
            if raw_request["stop"] is None:
                raw_request.pop("stop")

        # Special handling for gpt-4o-audio-preview
        # See: https://platform.openai.com/docs/guides/audio
        if model_engine.startswith("gpt-4o-audio-preview") or model_engine.startswith("gpt-4o-mini-audio-preview"):
            raw_request["modalities"] = ["text"]

            # Avoid error:
            # OpenAI error: Error code: 400 - {'error': {'message': "[{'type': 'string_type', 'loc': ('body', 'stop', 'str'), 'msg': 'Input should be a valid string', 'input': None}, {'type': 'list_type', 'loc': ('body', 'stop', 'list[str]'), 'msg': 'Input should be a valid list', 'input': None}, {'type': 'list_type', 'loc': ('body', 'stop', 'list[list[int]]'), 'msg': 'Input should be a valid list', 'input': None}]", 'type': 'invalid_request_error', 'param': None, 'code': None}}  # noqa: 3501
            if raw_request["stop"] is None:
                raw_request.pop("stop")

        def do_it() -> Dict[str, Any]:
            return self.client.chat.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            if self.INAPPROPRIATE_IMAGE_ERROR in str(e) or self.INAPPROPRIATE_PROMPT_ERROR in str(e):
                hlog(f"Failed safety check: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                return RequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    embedding=[],
                )
            elif self.OPENAI_SERVER_ERROR in str(e):
                # Handle these errors by returning an empty completion to unblock
                hlog(f"OpenAI server error for request: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.OPENAI_SERVER_ERROR},
                )
                return RequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    embedding=[],
                )
            elif self.INAPPROPRIATE_PROMPT_AZURE_ERROR in str(e) or self.INAPPROPRIATE_PROMPT_MICROSOFT_ERROR in str(e):
                return RequestResult(
                    success=False,
                    cached=False,
                    error="Content blocked by Azure's content management filter",
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                )

            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            # Handle Azure OpenAI content filter
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
            if raw_completion["finish_reason"] == "content_filter":
                hlog(f"Content blocked by OpenAI filter: {str(raw_request)}")
                return RequestResult(
                    success=False,
                    cached=False,
                    error="Content blocked by OpenAI filter",
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                )
            # The OpenAI chat completion API doesn't support echo.
            # If `echo_prompt` is true, combine the prompt and completion.
            raw_completion_content = raw_completion["message"]["content"]
            if self.output_processor:
                raw_completion_content = self.output_processor(raw_completion_content)
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
                logprob=0,  # OpenAI does not provide logprobs
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request: Dict[str, Any] = {
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo": request.echo_prompt,
        }

        # OpenAI doesn't let you ask for more completions than the number of
        # per-token candidates.
        raw_request["best_of"] = max(raw_request["best_of"], raw_request["n"])
        raw_request["logprobs"] = max(raw_request["logprobs"], raw_request["n"])

        return raw_request

    def _make_completion_request(self, request: Request) -> RequestResult:
        raw_request = self._to_raw_completion_request(request)

        def do_it() -> Dict[str, Any]:
            return self.client.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for (
                text,
                logprob,
            ) in zip(raw_data["tokens"], raw_data["token_logprobs"]):
                tokens.append(Token(text=text, logprob=logprob or 0))
                sequence_logprob += logprob or 0
            completion = GeneratedOutput(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            # OpenAI sends us back tokens past the end of text token,
            # so we need to manually truncate the list of tokens.
            # TODO: filed an issue with their support to check what the expected behavior here is.
            completion = truncate_sequence(
                completion, replace(request, stop_sequences=request.stop_sequences + [OpenAIClient.END_OF_TEXT])
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def _make_transcription_request(self, request: Request) -> RequestResult:
        assert (
            request.multimodal_prompt is not None and request.multimodal_prompt.size == 1
        ), "Expected just a single audio file."
        media_object = request.multimodal_prompt.media_objects[0]
        assert media_object.is_type("audio") and media_object.location, "Expected an audio file."
        audio_path: str = media_object.location
        model: str = self._get_model_for_request(request)

        def do_it() -> Dict[str, Any]:
            transcription = self.client.audio.transcriptions.create(model=model, file=open(audio_path, "rb"))
            return {"transcription": transcription.text}

        try:
            cache_key = self._get_cache_key({"audio": audio_path, "model": model}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[GeneratedOutput(text=response["transcription"], logprob=0, tokens=[])],
            embedding=[],
        )

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return self._make_embedding_request(request)
        elif "whisper" in request.model_engine:
            return self._make_transcription_request(request)
        else:
            return self._make_chat_request(request)


class OpenAILegacyCompletionsClient(OpenAIClient):
    def make_request(self, request: Request) -> RequestResult:
        return self._make_completion_request(request)


class OpenAITranscriptionThenCompletionClient(Client):
    """
    Wrapper around `OpenAIClient` that transcribes audio to text with a
    speech-to-text model (e.g., Whisper) before making a completion request.
    """

    @staticmethod
    def wrap_transcribed_indicator(transcription: str) -> str:
        return f"\n[TRANSCRIBED AUDIO START]\n{transcription}\n[TRANSCRIBED AUDIO END]\n"

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ):
        self._openai_client = OpenAIClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            org_id=org_id,
        )

    def make_request(self, request: Request) -> RequestResult:
        # Ensure that there is only one _ in the model engine name as the format is
        # `transcription-model_completion-model`
        assert request.model_engine.count("_") == 1, f"Invalid model name: {request.model_engine}"
        # Use `model_engine` to determine both the models for transcription and completion
        transcription_model, completion_model = request.model_engine.split("_")

        # Only multimodal prompts are supported
        assert request.multimodal_prompt is not None, "Expected a multimodal prompt"

        # Gather all the text content and transcribe any audio to text
        text_content: List[str] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("audio") and media_object.location:
                request = Request(
                    model=f"openai/{transcription_model}",
                    multimodal_prompt=MultimediaObject(media_objects=[media_object]),
                )
                response = self._openai_client.make_request(request)

                transcribed_text: str
                if response.success and response.completions:
                    transcribed_text = response.completions[0].text
                else:
                    transcribed_text = ""
                    hlog(f"Failed to transcribe audio: {response.error}")

                text_content.append(self.wrap_transcribed_indicator(transcribed_text))
            elif media_object.is_type(TEXT_TYPE):
                assert media_object.text is not None, "Expected text content"
                text_content.append(media_object.text)
            else:
                raise ValueError(f"Unrecognized media type: {media_object.type}")

        text_prompt: str = "\n".join(text_content)
        hlog(f"Transcribed prompt:\n{text_prompt}")

        # Now make the request to the completion model with just a text-only prompt and no audio
        # Use the same decoding parameters as the original request
        # Ensure to set multimodal_prompt to None so the request is treated as text-only.
        return self._openai_client.make_request(
            replace(request, prompt=text_prompt, model=f"openai/{completion_model}", multimodal_prompt=None)
        )
