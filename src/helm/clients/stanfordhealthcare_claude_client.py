from typing import Any, Dict, List

from helm.common.request import (
    Request,
    GeneratedOutput,
)
from helm.clients.stanfordhealthcare_http_model_client import StanfordHealthCareHTTPModelClient


class StanfordHealthCareClaudeClient(StanfordHealthCareHTTPModelClient):
    """
    Client for accessing Claude models hosted on Stanford Health Care's model API.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """

    def get_request(self, request: Request) -> Dict[str, Any]:
        return {
            "model_id": self.model,
            "prompt_text": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def parse_response(self, response: Dict[str, Any]) -> List[GeneratedOutput]:
        return [GeneratedOutput(text=item["text"], logprob=0, tokens=[]) for item in response["content"]]
