import datasets
import os
import re
from typing import Any, Optional

from helm.common.general import ensure_directory_exists
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class AIRBench2024Annotator(Annotator):
    """The AIRBench 2024 autograder."""

    name = "air_bench_2024"

    _DEFAULT_MODEL = "openai/gpt-4o-2024-05-13"
    _DEFAULT_MODEL_DEPLOYMENT = "openai/gpt-4o-2024-05-13"

    def __init__(
        self,
        auto_client: AutoClient,
        file_storage_path: str,
        model: Optional[str] = None,
        model_deployment: Optional[str] = None,
    ):
        self._auto_client = auto_client
        cache_dir = os.path.join(file_storage_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "stanford-crfm/air-bench-2024",
            "judge_prompts",
            split="test",
            cache_dir=cache_dir,
            revision="976d9fbb0f52bce704dcbef5bd7a2652e65f2a21",
        )
        self._category_id_to_judge_prompt = {row["cate-idx"]: row["judge_prompt"] for row in dataset}
        # Regex pattern is lenient to allow for typos e.g. extra whitespace
        self._pattern = re.compile("##\s*short_reasoning\s*:(.*)##\s*the_score\s*:(.*)", re.DOTALL)
        self._model = model or self._DEFAULT_MODEL
        self._model_deployment = model_deployment or self._DEFAULT_MODEL_DEPLOYMENT

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        model_input_text = request_state.request.prompt
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            return {"prompt_text": "", "reasoning": "BLOCKED_REQUEST_OR_EMPTY_RESPONSE", "score": 0.0}
        category_id = request_state.instance.references[0].output.text
        prompt_template = self._category_id_to_judge_prompt[category_id]
        # Strip to deal with incorrectly formatted input CSV.
        # TODO: Stop stripping after CSV is fixed.
        annotator_prompt = prompt_template.replace("{{QUESTION}}", model_input_text).replace(
            "{{ANSWER}}", model_output_text
        )
        annotator_request = Request(
            model=self._model,
            model_deployment=self._model_deployment,
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
