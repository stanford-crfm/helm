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


class CallCenterSummarizationPairwiseComparisonAnnotator(Annotator):
    """Annotator for call center summarization with pairwise comparison."""

    name = "call_center_summarization_pairwise_comparison"

    PROMPT_TEMPLATE = """\
    Given a call transcript and two different summaries of the call transcript, select your preferred summary, which can be subjective, considering the criteria below. Also provide a one-sentence reasoning for your selection.

    ### Criteria
    Faithfulness: Can all the information expressed by the summary can be inferred from the source?
    Relevance: To what extent the summary include only important information from the source?
    Coherence: Does the summary organize the relevant information into a well-structured summary?

    ### Call Transcript
    {{CALL_TRANSCRIPT}}

    ### Summary A
    {{SUMMARY_A}}

    ### Summary B
    {{SUMMARY_B}}

    ### Task
    Output only a JSON object with the following format:

    {"reasoning": "Reasoning", "selected": "A" or "B"}
    """  # noqa: E501

    def __init__(self, auto_client: AutoClient):
        super().__init__()
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        call_transcript = request_state.instance.input.text
        summary = request_state.result.completions[0].text.strip()
        assert len(request_state.instance.all_correct_references) == 1
        reference_summary = request_state.instance.all_correct_references[0].output.text
        if not summary.strip():
            hlog("Returning 0 scores due to empty response")
            return {"faithfulness": 0, "relevance": 0, "coherence": 0}
        assert request_state.instance.id is not None
        instance_id = int(request_state.instance.id[2:])
        if instance_id % 2:
            reference_option = "A"
            summary_a = reference_summary
            summary_b = summary
        else:
            reference_option = "B"
            summary_a = summary
            summary_b = reference_summary
        annotator_prompt = (
            textwrap.dedent(CallCenterSummarizationPairwiseComparisonAnnotator.PROMPT_TEMPLATE)
            .replace("{{CALL_TRANSCRIPT}}", call_transcript)
            .replace("{{SUMMARY_B}}", summary_a)
            .replace("{{SUMMARY_A}}", summary_b)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-08-06",
            model_deployment="openai/gpt-4o-2024-08-06",
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
        for expected_key in ["reasoning", "selected"]:
            if expected_key not in annotator_response_parsed:
                raise Exception(f"Malformed annotator response: {annotator_response_text}")
        score = 0.0
        selected = annotator_response_parsed["selected"].strip()
        if selected != "A" and selected != "B":
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        if selected == reference_option:
            score = 0.0
        else:
            score = 1.0
        return {
            "reasoning": annotator_response_parsed["reasoning"],
            "selected": selected,
            "reference_option": reference_option,
            "score": score,
        }


class CallCenterSummarizationKeyPointsRecallAnnotator(Annotator):
    """Annotator for call center summarization with key point recall."""

    name = "call_center_summarization_key_points_recall"

    PROMPT_TEMPLATE = """\
    Given a call transcript, a list of key points and a summary, determine which key points are included in the summary.

    ### Call Transcript
    {{CALL_TRANSCRIPT}}

    ### Key Points
    {{KEY_POINTS}}

    ### Summary
    {{SUMMARY}}

    ### Task
    Output only a JSON array of booleans, where each boolean indicates if the corresponding key point was included in the summary.
    """  # noqa: E501

    def __init__(self, auto_client: AutoClient):
        super().__init__()
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        call_transcript = request_state.instance.input.text
        summary = request_state.result.completions[0].text.strip()
        key_points = "\n".join(
            [f"- {reference.output.text}" for reference in request_state.instance.all_correct_references]
        )
        if not summary.strip():
            hlog("Returning 0 scores due to empty response")
            return {"faithfulness": 0, "relevance": 0, "coherence": 0}
        annotator_prompt = (
            textwrap.dedent(CallCenterSummarizationKeyPointsRecallAnnotator.PROMPT_TEMPLATE)
            .replace("{{CALL_TRANSCRIPT}}", call_transcript)
            .replace("{{KEY_POINTS}}", key_points)
            .replace("{{SUMMARY}}", summary)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-08-06",
            model_deployment="openai/gpt-4o-2024-08-06",
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
        # This strips everything outside the outermost [] brackets.
        json_start_index = annotator_response_text.find("[")
        json_end_index = annotator_response_text.rfind("]")
        if json_start_index < 0 or json_end_index < 0:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        annotator_response_json = annotator_response_text[json_start_index : json_end_index + 1]
        try:
            annotator_response_parsed = json.loads(annotator_response_json)
        except JSONDecodeError:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        if not len(annotator_response_parsed):
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        score = sum([1.0 if elem else 0.0 for elem in annotator_response_parsed]) / len(annotator_response_parsed)
        return {"key_points_found": json.dumps(annotator_response_parsed), "score": score}
