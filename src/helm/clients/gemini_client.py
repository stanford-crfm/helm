import json
import os
import requests
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, Optional, List, Union

from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Token, wrap_request_time, Request, RequestResult, GeneratedOutput, ErrorFlags
from helm.clients.client import CachingClient, truncate_sequence, generate_uid_for_multimodal_prompt

try:
    from google.generativeai import GenerativeModel, configure # Gemini
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google"])


_models_lock: Lock = Lock()
_models: Dict[str, Any] = {}


class GeminiContentBlockedError(Exception):
    pass


class GeminiClient(CachingClient, ABC):
    """Client for Gemini models"""

    def __init__(self, cache_config: CacheConfig, project_id: str, location: str) -> None:
        super().__init__(cache_config=cache_config)
        self.project_id = project_id
        self.location = location
        self.api_key = os.environ["GOOGLE_API_KEY"]
        

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        raise NotImplementedError

class GeminiChatClient(GeminiClient):
    """Client for Gemini chat models (e.g., Gemini). Supports multimodal prompts."""

    CONTENT_BLOCKED_FINISH_REASONS: List[int] = [
        3,  # SAFETY
        4,  # RECITATION
        6,  # BLOCKLIST
        7,  # PROHIBITED_CONTENT
        8,  # SPII (Sensitive Personally Identifiable Information)
    ]

    @staticmethod
    def get_model(model_name: str) -> GenerativeModel:
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
        }

        completions: List[GeneratedOutput] = []
        model_name: str = request.model_engine
        model = self.get_model(model_name)

        headers = {
            'content-type': 'application/json',
        }

        url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={self.api_key}"

        data = {
            'model':
            model_name,
            'contents': {
                "parts": [
                    {
                        "text": contents
                    }
                ]
            },
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
            ],
            'generationConfig': parameters
        }


        try:

            def do_it() -> Dict[str, Any]:

                raw_response = requests.post(url,
                                         headers=headers,
                                         data=json.dumps(data))

                response = raw_response.json()

                print(json.dumps(response, indent=2))


                candidates = response["candidates"]

                if not candidates:
                    raise GeminiContentBlockedError(f"No candidates in response: {response}")
                predictions: List[Dict[str, Any]] = []
                for candidate in candidates:
                    print(candidate)
                    # Depending on the version of the Gemini library and the type of prompt blocking,
                    # content blocking can show up in many ways, so this defensively handles most of these ways
                    if candidate["finishReason"] in GeminiChatClient.CONTENT_BLOCKED_FINISH_REASONS:
                        raise GeminiContentBlockedError(f"Content blocked with reason: {candidate['finishReason']}")
                    if not candidate.get("content"):
                        raise GeminiContentBlockedError(f"No content in candidate: {candidate}")
                    if not candidate["content"].get("parts"):
                        raise GeminiContentBlockedError(f"No content parts in candidate: {candidate}")
                    predictions.append({"text": "".join(map(lambda x: x["text"], candidate["content"]["parts"]))})
                    # TODO: Extract more information from the response
                return {"predictions": predictions, "output_tokens": int(response["usageMetadata"]["candidatesTokenCount"])}

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
        except GeminiContentBlockedError as e:
            return RequestResult(
                success=False,
                cached=False,
                error=f"Content blocked: {str(e)}",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            )
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"GeminiTextClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Handle cached responses with blocked content from old versions of HELM.
        if response["predictions"] is None:
            return RequestResult(
                success=False,
                cached=False,
                error=f"Content blocked error in cached response: {str(response)}",
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
                    error=f"Content blocked error in cached prediction: {str(prediction)}",
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
                    request_time=response["request_time"],
                    request_datetime=response["request_datetime"],
                )
            response_text = prediction["text"]

            # The Python SDK does not support echo

            tokens = [Token("a", 0) for _ in range(response["output_tokens"])]

            text: str = request.prompt + response_text if request.echo_prompt else response_text
            completion = GeneratedOutput(text=text, logprob=0, tokens=tokens)
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