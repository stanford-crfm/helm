from dataclasses import replace
import re
from typing import Any, Dict, List, Tuple

from helm.clients.vllm_client import VLLMChatClient
from helm.common.request import GeneratedOutput, Request, RequestResult, Thinking


class VLLMGraniteThinkingClient(VLLMChatClient):
    """Sends request to a Granite model on vLLM server with thinking enabled.

    From vLLM documentation at
    https://docs.vllm.ai/en/v0.9.1/features/reasoning_outputs.html

    IBM Granite 3.2 reasoning is disabled by default;
    to enable it, you must also pass thinking=True in your chat_template_kwargs.
    """

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
        return raw_request

    def _parse_thinking(self, input: str) -> Tuple[str, str]:
        """Return a tuple of thinking text and output text."""
        match = re.match(r"<think>(.*)</think>\s*<response>(.*)</response>", input, re.DOTALL)
        if match:
            return (match.group(1), match.group(2))

        match = re.match(r"<think>(.*)</think>\s*<response>(.*)", input, re.DOTALL)
        if match:
            return (match.group(1), match.group(2))

        match = re.match(r"<think>(.*)</think>\s*", input, re.DOTALL)
        if match:
            return (match.group(1), "")

        match = re.match(r"<think>(.*)", input, re.DOTALL)
        if match:
            return (match.group(1), "")

        return (input, "")

    def _make_chat_request(self, request: Request) -> RequestResult:
        request_result = super()._make_chat_request(request)
        modified_completions: List[GeneratedOutput] = []
        for completion in request_result.completions:
            thinking, modified_text = self._parse_thinking(completion.text)
            modified_completions.append(
                replace(
                    completion,
                    text=modified_text,
                    thinking=Thinking(text=thinking),
                )
            )
        return replace(request_result, completions=modified_completions)
