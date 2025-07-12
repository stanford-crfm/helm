from typing import Any, Optional, Dict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


PROMPT_TEMPLATE = """**Task:** Determine if an answer to the question is supported by a given text.

**Input (in {target_language}):**
- Text
- Question
- Answer

**Single Word Output (in English):**
- YES: Answer is derived from the text.
- NO: Answer is not derived from the text.

Text:
{context}

Question:
{question}

Answer:
{predicted_answer}

Output:"""


class EclekticAnnotator(Annotator):
    """The Eclektic autograder."""

    name = "eclektic_autograder"

    _DEFAULT_MODEL = "google/gemini-2.5-pro"
    _DEFAULT_MODEL_DEPLOYMENT = "google/gemini-2.5-pro"

    def __init__(
        self,
        auto_client: AutoClient,
        file_storage_path: str,
        model: Optional[str] = None,
        model_deployment: Optional[str] = None,
    ):
        self._auto_client = auto_client

        self._model = model or self._DEFAULT_MODEL
        self._model_deployment = model_deployment or self._DEFAULT_MODEL_DEPLOYMENT

    def annotate(self, request_state: RequestState) -> Any:

        assert request_state.result
        assert len(request_state.result.completions) == 1

        model_input_text = request_state.request.prompt
        model_output_text = request_state.result.completions[0].text
        extra_data: Dict[str, Any] = request_state.instance.extra_data or {}
        language: str = str(extra_data.get("lang", ""))
        context: str = str(extra_data.get("content", ""))

        annotator_prompt = [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    target_language=language,
                    context=context,
                    question=model_input_text,
                    predicted_answer=model_output_text,
                ),
            }
        ]

        annotator_request = Request(
            model=self._model,
            model_deployment=self._model_deployment,
            messages=annotator_prompt,
            temperature=0.0,
            max_tokens=5,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        postprocess_annotator_response = self._postprocess(annotator_response_text)

        if not postprocess_annotator_response:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")

        return postprocess_annotator_response

    def _postprocess(self, output):
        result = {}
        if "yes" in output.lower():
            eval_response = 1
        else:
            eval_response = 0
        result["correct"] = eval_response
        return result
