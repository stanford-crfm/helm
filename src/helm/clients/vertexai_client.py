import requests
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, Optional, List, Union

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, ErrorFlags
from helm.clients.client import CachingClient, truncate_sequence, generate_uid_for_multimodal_prompt

try:
    import vertexai
    from vertexai.language_models import TextGenerationModel, TextGenerationResponse  # PaLM2
    from vertexai.preview.generative_models import GenerativeModel, GenerationResponse, Candidate, Part, Image  # Gemini
    from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google"])


_models_lock: Lock = Lock()
_models: Dict[str, Any] = {}


class VertexAIContentBlockedError(Exception):
    pass


class VertexAIClient(CachingClient, ABC):
    """Client for Vertex AI models"""

    def __init__(self, cache_config: CacheConfig, project_id: str, location: str) -> None:
        super().__init__(cache_config=cache_config)
        self.project_id = project_id
        self.location = location

        # VertexAI's default safety filter is overly sensitive, so we disable it.
        self.safety_settings: Dict[HarmCategory, SafetySetting.HarmBlockThreshold] = {
            harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
            for harm_category in iter(HarmCategory)
        }

        vertexai.init(project=self.project_id, location=self.location)

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        raise NotImplementedError


class VertexAITextClient(VertexAIClient):
    """Client for Vertex AI text models
    This client is used for PaLM2 for example."""

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        parameters = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            "stop_sequences": request.stop_sequences,
            "candidate_count": request.num_completions,
            # TODO #2084: Add support for these parameters.
            # The parameters "echo", "frequency_penalty", and "presence_penalty" are supposed to be supported
            # in an HTTP request (See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text),
            # but they are not supported in the Python SDK:
            # https://github.com/googleapis/python-aiplatform/blob/beae48f63e40ea171c3f1625164569e7311b8e5a/vertexai/language_models/_language_models.py#L968C1-L980C1
            # "frequency_penalty": request.frequency_penalty,
            # "presence_penalty": request.presence_penalty,
            # "echo": request.echo_prompt,
        }

        completions: List[GeneratedOutput] = []
        model_name: str = request.model_engine

        try:

            def do_it() -> Dict[str, Any]:
                model = TextGenerationModel.from_pretrained(model_name)
                response = model.predict(request.prompt, **parameters)
                candidates: List[TextGenerationResponse] = response.candidates
                response_dict = {
                    "predictions": [{"text": completion.text for completion in candidates}],
                }  # TODO: Extract more information from the response
                return response_dict

            # We need to include the engine's name to differentiate among requests made for different model
            # engines since the engine name is not included in the request itself.
            # Same for the prompt.
            cache_key = CachingClient.make_cache_key(
                {
                    "engine": request.model_engine,
                    "prompt": request.prompt,
                    **parameters,
                },
                request,
            )

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"VertexAITextClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        for prediction in response["predictions"]:
            response_text = prediction["text"]

            # The Python SDK does not support echo
            text: str = request.prompt + response_text if request.echo_prompt else response_text

            # TODO #2085: Add support for log probs.
            # Once again, log probs seem to be supported by the API but not by the Python SDK.
            # HTTP Response body reference:
            # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text#response_body
            # Python SDK reference:
            # https://github.com/googleapis/python-aiplatform/blob/beae48f63e40ea171c3f1625164569e7311b8e5a/vertexai/language_models/_language_models.py#L868
            completion = GeneratedOutput(text=text, logprob=0, tokens=[])
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


class VertexAIChatClient(VertexAIClient):
    """Client for Vertex AI chat models (e.g., Gemini). Supports multimodal prompts."""

    # Set the finish reason to this if the prompt violates the content policy
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = "The prompt violates Google's content policy."

    # Gemini returns this error for certain valid requests
    CONTENT_HAS_NO_PARTS_ERROR: str = "Content has no parts."

    # Enum taken from:
    # https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#google.cloud.aiplatform.v1beta1.Candidate.FinishReason
    # We don't directly import this enum because it can differ between different Vertex AI library versions.
    CONTENT_BLOCKED_FINISH_REASONS: List[int] = [
        3,  # SAFETY
        4,  # RECITATION
        6,  # BLOCKLIST
        7,  # PROHIBITED_CONTENT
        8,  # SPII (Sensitive Personally Identifiable Information)
    ]

    @staticmethod
    def get_model(model_name: str) -> Any:
        global _models_lock
        global _models

        with _models_lock:
            if model_name not in _models:
                _models[model_name] = GenerativeModel(model_name)
            return _models[model_name]

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        contents: str = request.prompt

        # For the multimodal case, build up the content with the media objects of `request.multimodal_prompt`
        if request.multimodal_prompt is not None:
            return self._make_multimodal_request(request)

        parameters = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            "stop_sequences": request.stop_sequences,
            "candidate_count": request.num_completions,
            # TODO #2084: Add support for these parameters.
            # The parameters "echo", "frequency_penalty", and "presence_penalty" are supposed to be supported
            # in an HTTP request (See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text),
            # but they are not supported in the Python SDK:
            # https://github.com/googleapis/python-aiplatform/blob/beae48f63e40ea171c3f1625164569e7311b8e5a/vertexai/language_models/_language_models.py#L968C1-L980C1
            # "frequency_penalty": request.frequency_penalty,
            # "presence_penalty": request.presence_penalty,
            # "echo": request.echo_prompt,
        }

        completions: List[GeneratedOutput] = []
        model_name: str = request.model_engine
        model = self.get_model(model_name)

        try:

            def do_it() -> Dict[str, Any]:
                # Here we differ from Vertex AI's tutorial.
                # https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-chat-prompts-gemini#send_chat_prompts   # noqa: E501
                # It would advise to use model.start_chat() but since we do not want to use Chat capabilities of
                # Vertex AI, we use model.generate_text() instead. Furthermore, chat.send_message() restricts the
                # output to only one candidate.
                # chat: ChatSession = model.start_chat()
                # See: https://github.com/googleapis/python-aiplatform/blob/e8c505751b10a9dc91ae2e0d6d13742d2abf945c/vertexai/generative_models/_generative_models.py#L812  # noqa: E501
                response: GenerationResponse = model.generate_content(
                    contents, generation_config=parameters, safety_settings=self.safety_settings
                )
                candidates: List[Candidate] = response.candidates

                # Depending on the version of the Vertex AI library and the type of content blocking,
                # content blocking can show up in many ways, so this defensively handles most of these ways
                if not candidates:
                    raise VertexAIContentBlockedError("No candidates in response due to content blocking")
                predictions: List[Dict[str, Any]] = []
                for candidate in candidates:
                    if (
                        candidate.finish_reason in VertexAIChatClient.CONTENT_BLOCKED_FINISH_REASONS
                        or not candidate.content.parts
                    ):
                        # The prediction was either blocked due to safety settings or the model stopped and returned
                        # nothing (which also happens when the model is blocked).
                        # For now, we don't cache blocked requests, because we are trying to get the
                        # content blocking removed.
                        raise VertexAIContentBlockedError("Content has no parts due to content blocking")
                    predictions.append({"text": candidate.content.text})
                    # TODO: Extract more information from the response
                return {"predictions": predictions}

            # We need to include the engine's name to differentiate among requests made for different model
            # engines since the engine name is not included in the request itself.
            # Same for the prompt.
            cache_key = CachingClient.make_cache_key(
                {
                    "model_name": model_name,
                    "prompt": request.prompt,
                    **parameters,
                },
                request,
            )

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except VertexAIContentBlockedError:
            return RequestResult(
                success=False,
                cached=False,
                error="Response was empty due to content moderation filter",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            )
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"VertexAITextClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Handle cached responses with blocked content from old versions of HELM.
        if response["predictions"] is None:
            return RequestResult(
                success=False,
                cached=False,
                error="Response was empty due to content moderation filter",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                request_time=response["request_time"],
                request_datetime=response["request_datetime"],
            )

        for prediction in response["predictions"]:
            # Handle cached responses with blocked content from old versions of HELM.
            if "text" not in prediction:
                return RequestResult(
                    success=False,
                    cached=False,
                    error="Response was empty due to content moderation filter",
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                    request_time=response["request_time"],
                    request_datetime=response["request_datetime"],
                )
            response_text = prediction["text"]

            # The Python SDK does not support echo
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            completion = GeneratedOutput(text=text, logprob=0, tokens=[])
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

    def _make_multimodal_request(self, request: Request) -> RequestResult:
        def complete_for_valid_error(error_message: str) -> RequestResult:
            empty_completion = GeneratedOutput(
                text="",
                logprob=0,
                tokens=[],
                finish_reason={"reason": error_message},
            )
            return RequestResult(
                success=True,
                cached=False,
                request_time=0,
                completions=[empty_completion] * request.num_completions,
                embedding=[],
            )

        # Contents can either be text or a list of multimodal content made up of text, images or other content
        contents: Union[str, List[Union[str, Any]]] = request.prompt
        # Used to generate a unique cache key for this specific request
        assert request.multimodal_prompt is not None
        prompt_key: str = generate_uid_for_multimodal_prompt(request.multimodal_prompt)

        # For the multimodal case, build up the content with the media objects of `request.multimodal_prompt`
        contents = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                contents.append(Part.from_image(Image.load_from_file(media_object.location)))
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                contents.append(media_object.text)
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        parameters = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            "stop_sequences": request.stop_sequences,
            "candidate_count": 1,
        }

        completions: List[GeneratedOutput] = []
        model_name: str = request.model_engine
        model = self.get_model(model_name)

        request_time = 0
        request_datetime: Optional[int] = None
        all_cached = True

        # Gemini Vision only supports generating 1-2 candidates at a time, so make `request.num_completions` requests
        for completion_index in range(request.num_completions):
            try:

                def do_it() -> Dict[str, Any]:
                    raw_response = model.generate_content(
                        contents, generation_config=parameters, safety_settings=self.safety_settings
                    )
                    if raw_response._raw_response.prompt_feedback.block_reason != 0:
                        hlog(f"Content blocked for prompt: {request.multimodal_prompt}")
                        return {"error": self.CONTENT_POLICY_VIOLATED_FINISH_REASON}

                    return {"predictions": [{"text": raw_response.candidates[0].text}]}

                raw_cache_key = {"model_name": model_name, "prompt": prompt_key, **parameters}
                if completion_index > 0:
                    raw_cache_key["completion_index"] = completion_index

                cache_key = CachingClient.make_cache_key(raw_cache_key, request)
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, ValueError) as e:
                if str(e) == self.CONTENT_HAS_NO_PARTS_ERROR:
                    return complete_for_valid_error(self.CONTENT_HAS_NO_PARTS_ERROR)

                error: str = f"Gemini Vision error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            if "error" in response:
                return complete_for_valid_error(response["error"])

            response_text = response["predictions"][0]["text"]
            completion = GeneratedOutput(text=response_text, logprob=0, tokens=[])
            sequence = truncate_sequence(completion, request, print_warning=True)
            completions.append(sequence)

            request_time += response["request_time"]
            # Use the datetime from the first completion because that's when the request was fired
            request_datetime = request_datetime or response.get("request_datetime")
            all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
