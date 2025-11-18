import re
from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException


class ArabicToolUsageAnnotator(Annotator):
    name = "arabic_tool_usage"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        with open("/home/yifanmai/oss/helm/arabic-bench/tool_usage/Eval_prompt.txt", "r") as f:
            self._template = f.read()

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        question_text = request_state.instance.input.text
        reference_text = request_state.instance.references[0].output.text
        prediction_text = request_state.result.completions[0].text

        annotator_prompt = (
            self._template.replace("{query}", question_text)
            .replace("{gold_tools}", reference_text)
            .replace("{model_answer}", prediction_text)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=100,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        score = annotator_response_text.strip()

        return {"prompt": annotator_prompt, "score": score}
