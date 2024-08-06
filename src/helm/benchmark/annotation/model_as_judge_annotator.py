import json
import os
from typing import Dict

from helm.common.general import ensure_directory_exists
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class ModelAsJudgeAnnotator(Annotator):
    """Model as judge autograder."""

    def __init__(self, auto_client: AutoClient, file_storage_path: str):
        self._auto_client = auto_client
        cache_dir = os.path.join(file_storage_path, "data")
        ensure_directory_exists(cache_dir)
        super().__init__()

    # expects a prompt that includes a user request that ensures the model needs to return a score and reasoning
    def score_with_reasoning(
        self,
        annotator_prompt: str,
        annotator_model: str,
        annotator_model_deployment: str,
    ) -> Dict:
        annotator_request = Request(
            model=annotator_model,
            model_deployment=annotator_model_deployment,
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=256,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        json_start_index = annotator_response_text.find("{")
        json_end_index = annotator_response_text.rfind("}")
        if json_start_index < 0 or json_end_index < 0:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        annotator_response_json = annotator_response_text[json_start_index : json_end_index + 1]
        try:
            parsed_response = json.loads(annotator_response_json)
        except Exception as e:
            raise Exception(f"Malformed annotator response: {annotator_response_text}") from e

        if not parsed_response:
            raise Exception(f"Malformed annotator response: {annotator_response_text}")
        reasoning = parsed_response.get("reasoning", "").strip()
        try:
            score = float(parsed_response.get("score", ""))
        except ValueError as e:
            raise Exception(f"Malformed annotator response: {annotator_response_text}") from e

        return {"reasoning": reasoning, "score": score}
