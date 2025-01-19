from typing import Any, Dict, List

from helm.common.request import (
    Request,
    GeneratedOutput,
)
from helm.clients.stanfordhealthcare_http_model_client import StanfordHealthCareHTTPModelClient


class StanfordHealthCareGoogleClient(StanfordHealthCareHTTPModelClient):
    """
    Client for accessing Google models hosted on Stanford Health Care's model API.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """
    
    def get_request(self, request: Request) -> Dict[str, Any]:
        return {
            "contents": {
                "role": "user",
                "parts": {"text": request.prompt},
            },
            "safety_settings": {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            "generation_config": {
                "temperature": request.temperature,
                "topP": request.top,
                "topK": request.top_k_per_token,
                "candidateCount": 1,
                "maxOutputTokens": request.max_tokens,
                "stopSequences": request.stop_sequences,
            },
        }

    def parse_response(self, response: Dict[str, Any]) -> List[GeneratedOutput]:
        completions = []
        for item in response:
            for candidate in item.get("candidates", []):
                text_parts = "".join(part["text"] for part in candidate.get("content", {}).get("parts", []))
                completions.append(GeneratedOutput(text=text_parts, logprob=0, tokens=[]))
        return completions
