import os
import re
from typing import Any

from helm.common.general import ensure_directory_exists
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class MedicationQAAnnotator(Annotator):
    """The LiveQA autograder."""

    name = "medication_qa"

    def __init__(self, auto_client: AutoClient, file_storage_path: str):
        self._auto_client = auto_client
        cache_dir = os.path.join(file_storage_path, "data")
        ensure_directory_exists(cache_dir)
        # Regex pattern is lenient to allow for typos e.g. extra whitespace
        self._pattern = re.compile("##\s*short_reasoning\s*:(.*)##\s*the_score\s*:(.*)", re.DOTALL)

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        model_input_text = request_state.request.prompt
        model_output_text = request_state.result.completions[0].text
        correct_response = request_state.instance.references[0].output.text
        prompt_template = """You are a medical judge will be given a user request and a model response to that request. Your task is to evaluate whether the model response properly answers the request based on a set of correct answers.
The user's request will be provided in these tags:
<user_request>
{{QUESTION}}
</user_request>
The model's response will be provided in these tags:
<model_response>
{{ANSWER}}
</model_response>
Some potential correct responses will be provided in these tags:
<correct_response>
{{CORRECT_RESPONSE}}
</correct_response>
Carefully analyze the <model_response> and determine if the model is correct with respect to one or more of the <correct_responses> on a range from 0 to 1.
# 1, if the model response fully matches the correct response content and intent for at least one correct response
# 0.5, if the model response contains some correct information but is incomplete and/or contains incorrect information relative to the correct response(s)
# 0, if the model response is completely unrelated or incorrect relative to the correct response(s)
Output your one-sentence concise reasoning in the ##short_reasoning tag and your score in the ##the_score tag. Your reasoning should be less than 20 tokens. The score should be a single number with no other output.
Format:
##short_reasoning: <your concise and brief reasoning>
##the_score: <your score>"""  # noqa: E501
        annotator_prompt = (
            prompt_template.strip('"')
            .strip()
            .replace("{{QUESTION}}", model_input_text)
            .replace("{{ANSWER}}", model_output_text)
            .replace("{{CORRECT_RESPONSE}}", correct_response)
        )
        annotator_request = Request(
            model="openai/gpt-4-turbo-2024-04-09",
            model_deployment="openai/gpt-4-turbo-2024-04-09",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=64,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        annotator_response_parts = self._pattern.search(annotator_response_text)
        if not annotator_response_parts:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        reasoning = annotator_response_parts[1].strip()
        try:
            score = float(annotator_response_parts[2].strip())
        except ValueError as e:
            raise Exception(f"Malformed annotator response: {annotator_response_text}") from e

        return {"prompt_text": annotator_prompt, "reasoning": reasoning, "score": score}
