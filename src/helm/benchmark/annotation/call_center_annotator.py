import json
from json.decoder import JSONDecodeError
import textwrap
from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request


class CallCenterSummarizationAnnotator(Annotator):
    """Annotator for call center summarization."""

    name = "call_center_summarization"

    PROMPT_TEMPLATE = """\
    Score the summary by scoring it on a five-point scale between 1 and 5 using three criteria: Faithfulness, Relevance, and Coherence.

    ### Criteria
    Faithfulness: Can all the information expressed by the summary can be inferred from the source? (1 = not at all, 5 = very much)
    Relevance: To what extent the summary include only important information from the source? (1 = not at all, 5 = very much)
    Coherence: Does the summary organize the relevant information into a well-structured summary? (1 = not at all, 5 = very much)

    ### Call Transcript
    {{CALL_TRANSCRIPT}}

    ### Summary
    {{SUMMARY}}

    ### Task
    Respond with only a raw JSON object in the following format, without using Markdown formatting:

    {"faithfulness": <score>, "relevance": <score>, "coherence": <score>}
    """  # noqa: E501

    CRITERIA = [
        "faithfulness",
        "relevance",
        "coherence",
    ]

    def __init__(self, auto_client: AutoClient):
        super().__init__()
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        call_transcript = request_state.instance.input.text
        summary = request_state.result.completions[0].text.strip()
        if not summary.strip():
            hlog("Returning 0 scores due to empty response")
            return {"faithfulness": 0, "relevance": 0, "coherence": 0}
        annotator_prompt = (
            textwrap.dedent(CallCenterSummarizationAnnotator.PROMPT_TEMPLATE)
            .replace("{{CALL_TRANSCRIPT}}", call_transcript)
            .replace("{{SUMMARY}}", summary)
        )
        annotator_request = Request(
            model="openai/gpt-4o-mini-2024-07-18",
            model_deployment="openai/gpt-4o-mini-2024-07-18",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=256,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        # OpenAI models like to surround JSON objects with ```json and ``` Markdown formatting.
        # This strips everything outside the outermost {} brackets.
        json_start_index = annotator_response_text.find("{")
        json_end_index = annotator_response_text.rfind("}")
        if json_start_index < 0 or json_end_index < 0:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        annotator_response_json = annotator_response_text[json_start_index : json_end_index + 1]
        try:
            annotator_response_parsed = json.loads(annotator_response_json)
        except JSONDecodeError:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        for expected_key in CallCenterSummarizationAnnotator.CRITERIA:
            if expected_key not in annotator_response_parsed:
                raise Exception(f"Malformed annotator response: {annotator_response_text}")
        return annotator_response_parsed
