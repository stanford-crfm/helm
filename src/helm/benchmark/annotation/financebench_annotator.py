import json
from json.decoder import JSONDecodeError
from typing import Any

from helm.benchmark.scenarios.scenario import CORRECT_TAG
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class FinanceBenchAnnotator(Annotator):
    """Annoator for FinanceBench that uses GPT-4o to determine if the model response is correct."""

    name = "financebench"
    _PROMPT_TEMPLATE = """Classify the model's response as one of three categories: "correct_answer", "incorrect_answer", or "failure_to_answer". Additionally, provide a short, one-sentence explanation for your classification.

Categories:

correct_answer: Allow minor deviations, such as giving the answer in billions when the unit was given in the question as millions.

incorrect_answer: This includes calculations that are off by small margins to several orders of magnitude, and from making up legal information to giving the wrong direction for an effect (e.g. reporting negative growth when it is actually positive). If a model gives the right answer but with logic or calculations that explicitly contradict the evidence in the gold standard answer, label it as incorrect_answer.

failure_to_answer: If the model explicitly states that it cannot answer because it does not have access to the right information then it is a failure to answer.

Question: {{QUESTION}}

Gold answer: {{GOLD_ANSWER}}

Model's response: {{MODEL_RESPONSE}}

Respond with only a raw JSON object in the following format, without using Markdown formatting:

{"explanation": "<one sentence explanation>", "label": "<category>"}
"""  # noqa: E501

    def __init__(self, auto_client: AutoClient, file_storage_path: str):
        super().__init__()
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert len(request_state.instance.references[0].tags) == 1
        assert request_state.instance.references[0].tags[0] == CORRECT_TAG
        question = request_state.instance.input.text.split("\nQuestion: ")[-1].strip()
        gold_answer = request_state.instance.references[0].output.text.strip()
        model_response = request_state.result.completions[0].text.strip()
        if not model_response.strip():
            return {"reasoning": "BLOCKED_REQUEST_OR_EMPTY_RESPONSE", "label": "failure_to_answer"}
        annotator_prompt = (
            FinanceBenchAnnotator._PROMPT_TEMPLATE.replace("{{QUESTION}}", question)
            .replace("{{GOLD_ANSWER}}", gold_answer)
            .replace("{{MODEL_RESPONSE}}", model_response)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=64,
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
        return annotator_response_parsed
