import base64
import dataclasses
import requests
from typing import Any, Dict, List, Union, Optional

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.caching_tokenizer import CachingTokenizer
from helm.proxy.retry import NonRetriableException

try:
    import google.auth
    import google.auth.transport.requests
    from google.auth.exceptions import DefaultCredentialsError
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google"])


class VertexAIAuthenticationException(NonRetriableException):
    pass


class VertexAITokenizer(CachingTokenizer):
    """Google Vertex AI API for tokenization.

    Doc: https://cloud.google.com/vertex-ai/docs/generative-ai/compute-token"""

    def __init__(self, project_id: Optional[str], location: Optional[str], cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        if not project_id:
            raise VertexAIAuthenticationException("credentials.conf is missing googleProjectId")
        if not location:
            raise VertexAIAuthenticationException("credentials.conf is missing googleLocation")
        self.project_id = project_id
        self.location = location
        try:
            creds, _ = google.auth.default(quota_project_id=self.project_id)
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
        except DefaultCredentialsError as e:
            raise VertexAIAuthenticationException(
                "Log in using `gcloud auth application-default login` to use the Google Vertex tokenizer API"
            ) from e
        self.access_token = creds.token

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        cache_key = dataclasses.asdict(request)
        # Delete encode because the Google Vertex AI API simulateously gives string and integer tokens.
        del cache_key["encode"]
        return cache_key

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text: str = request["text"]
        tokenizer_name = request["tokenizer"].split("/", maxsplit=1)[1]
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/google/models/{tokenizer_name}:computeTokens"
        )

        headers = {"Authorization": f"Bearer {self.access_token}"}
        body = {
            "instances": [{"prompt": text}],
        }
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        tokens: List[Union[int, str]]
        response_instance = response["tokensInfo"][0]
        if not response_instance:
            # Response was empty
            tokens = []
        else:
            if request.encode:
                tokens = [int(token) for token in response_instance["tokenIds"]]
            else:
                # errors="ignore" is needed because the tokenizer is not guaranteed to tokenize on
                # the boundary of UTF-8 characters. The tokenization boundary can be within the bytes of
                # a UTF-8 character.
                #
                # TODO(#2141): Come up with a more correct way of doing this.
                tokens = [
                    base64.decodebytes(token.encode()).decode("utf-8", errors="ignore")
                    for token in response_instance["tokens"]
                ]
        return [TokenizationToken(token) for token in tokens]

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Defined for mypy but decode() already raises NotImplementedError
        raise NotImplementedError("The Google Vertex AI API does not support decoding.")
