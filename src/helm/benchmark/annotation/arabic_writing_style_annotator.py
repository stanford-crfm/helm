from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException


class AnnotatorResponseParseFailure(NonRetriableException):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


class ArabicWritingStyleAnnotator(Annotator):
    name = "arabic_writing_style"

    _PROMPT_FILE_PATH = "/home/yifanmai/oss/helm/arabic-bench/writing_style/Eval_prompt.txt"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        with open(self._PROMPT_FILE_PATH, "r") as f:
            self._prompt_template = f.read()

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        question_text = request_state.instance.input.text
        reference_text = question_text.split("\n\n", maxsplit=1)[-1].split(":", maxsplit=1)[-1].lstrip()
        prediction_text = request_state.result.completions[0].text

        annotator_prompt = self._prompt_template.replace("[REFERENCE_TEXT]", reference_text).replace(
            "[CANDIDATE_TEXT]", prediction_text
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
        score = float(annotator_response.completions[0].text) / 10.0

        return {"score": score}
