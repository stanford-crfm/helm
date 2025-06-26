from typing import Any, Dict

from helm.clients.vllm_client import VLLMClient
from helm.common.request import Request


class VLLMGraniteThinkingClient(VLLMClient):
    """Sends request to a Granite model on vLLM server with thinking enabled.

    From vLLM documentation at
    https://docs.vllm.ai/en/v0.9.1/features/reasoning_outputs.html

    IBM Granite 3.2 reasoning is disabled by default;
    to enable it, you must also pass thinking=True in your chat_template_kwargs.
    """

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["chat_template_kwargs"] = {"chat_template_kwargs": {"thinking": True}}
        return raw_request
