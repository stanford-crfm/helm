from typing import Dict, List
import json

import requests

from helm.common.cache import CacheConfig
from helm.common.images_utils import encode_base64
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt, truncate_and_tokenize_response_text
from helm.tokenizers.tokenizer import Tokenizer


class PalmyraVisionClient(CachingClient):
    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, endpoint: str, cache_config: CacheConfig):
        super().__init__(cache_config)
        self.tokenizer: Tokenizer = tokenizer
        self.tokenizer_name: str = tokenizer_name

        # Currently, the Palmyra Vision model does not have a public API, so we need to use a secret endpoint
        self.endpoint: str = endpoint

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Build the prompt
        prompt: List[Dict[str, str]] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                prompt.append(
                    {
                        "type": "InlineData",
                        "value": encode_base64(media_object.location, format="JPEG"),
                        "contentType": "image/jpeg",
                    }
                )
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                prompt.append({"type": "Text", "value": media_object.text})
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        # Generate
        try:

            def do_it():
                response = requests.post(
                    self.endpoint, headers={"Content-Type": "application/json"}, data=json.dumps({"parts": prompt})
                )
                assert response.status_code == 200, f"Got status code {response.status_code}: {response.text}"
                json_response = json.loads(response.text)
                assert (
                    "choices" in json_response and "errors" not in json_response
                ), f"Invalid response: {response.text}"
                return json_response

            cache_key = CachingClient.make_cache_key(
                raw_request={"prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt)},
                request=request,
            )
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as ex:
            return RequestResult(success=False, cached=False, error=str(ex), completions=[], embedding=[])

        # The internal endpoint doesn't support any other parameters, so we have to truncate ourselves
        completions: List[GeneratedOutput] = [
            truncate_and_tokenize_response_text(choice["text"], request, self.tokenizer, self.tokenizer_name)
            for choice in result["choices"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
