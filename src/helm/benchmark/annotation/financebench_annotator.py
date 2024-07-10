import datasets
import os
import re
import json
from typing import Any

from helm.benchmark.scenarios.scenario import CORRECT_TAG
from helm.common.general import ensure_directory_exists
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class FinanceBenchAnnotator(Annotator):
    """FinanceBenchAnnotator."""

    name = "financebench"
    _PROMPT_TEMPLATE = """Classify the model's response as one of three categories: "correct_answer", "incorrect_answer", or "failure_to_answer".

correct_answer: Allow minor deviations, such as giving the answer in billions when the unit was given in the question as millions.

incorrect_answer: This includes calculations that are off by small margins to several orders of magnitude, and from making up legal information to giving the wrong direction for an effect (e.g. reporting negative growth when it is actually positive). If a model gives the right answer but with logic or calculations that explicitly contradict the evidence in the gold standard answer, label it as incorrect_answer.

failure_to_answer: If the model explicitly states that it cannot answer because it does not have access to the right information then it is a failure to answer.

Additionally, provide a short, one-sentence explanation for your classification. Respond in the following JSON format (do not use Markdown):

{"explanation": "<one sentence explanation>", "label": "<classification>"}

Question: {{QUESTION}}

Gold answer: {{GOLD_ANSWER}}

Model response: {{MODEL_RESPONSE}}"""

    def __init__(self, auto_client: AutoClient, file_storage_path: str):
        super().__init__()
        self._auto_client = auto_client
        # cache_dir = os.path.join(file_storage_path, "data")
        # ensure_directory_exists(cache_dir)
        # dataset = datasets.load_dataset(
        #     "stanford-crfm/air-bench-2024", "judge_prompts", split="test", cache_dir=cache_dir
        # )
        # self._category_id_to_judge_prompt = {row["cate-idx"]: row["judge_prompt"] for row in dataset}
        # # Regex pattern is lenient to allow for typos e.g. extra whitespace
        # self._pattern = re.compile("##\s*short_reasoning\s*:(.*)##\s*the_score\s*:(.*)", re.DOTALL)

    def annotate(self, request_state: RequestState) -> Any:

        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert len(request_state.instance.references[0].tags) == 1
        assert request_state.instance.references[0].tags[0] == CORRECT_TAG
        question = request_state.instance.input.text
        gold_answer = request_state.instance.references[0].output.text
        model_response = request_state.result.completions[0].text
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
        try:
            annotator_response_parsed = json.loads(annotator_response_text)
        except:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        return annotator_response_parsed
