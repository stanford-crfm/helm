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
            "generation_config": {
                "temperature": request.temperature,
                "topP": request.top_p,
                "topK": request.top_k_per_token,
                "candidateCount": 1,
                "maxOutputTokens": request.max_tokens,
                "stopSequences": request.stop_sequences,
            },
        }

    def parse_response(self, response: Dict[str, Any]) -> List[GeneratedOutput]:
        completion = ""
        for item in response["content"]:
            if "content" in item["candidates"][0]:
                completion += item["candidates"][0]["content"]["parts"][0]["text"]
        return [GeneratedOutput(text=completion, logprob=0, tokens=[])]
