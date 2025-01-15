import re
from typing import Any
from importlib.resources import files

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class WildBenchAnnotator(Annotator):
    """The WildBench autograder."""

    name = "wildbench"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        template_path = files("helm.benchmark.annotation.wildbench").joinpath("eval_template.score.v2.md")
        with template_path.open("r") as f:
            self._score_template = f.read()
        self._pattern = re.compile(
            r'"strengths"\s*:\s*"(.*?)"\s*,\s*"weaknesses"\s*:\s*"(.*?)"\s*,\s*"score"\s*:\s*(".*?"|\d+)', re.DOTALL
        )

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert request_state.instance.extra_data
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            # Following https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/src/eval.py
            return {"prompt_text": "", "strengths": "N/A", "weaknesses": "The model output is empty.", "score": 1.0}
        prompt_template = self._score_template

        annotator_prompt = (
            prompt_template.replace("{$history}", request_state.instance.extra_data["history"])
            .replace("{$user_query}", request_state.instance.extra_data["user_query"])
            .replace("{$model_output}", model_output_text)
            .replace("{$checklist}", "\n".join(request_state.instance.extra_data["checklist"]))
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=2000,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        annotator_response_parts = self._pattern.search(annotator_response_text)
        if not annotator_response_parts:
            raise ValueError(f"Malformed annotator response: {annotator_response_text}")

        strengths = annotator_response_parts[1].strip()
        weaknesses = annotator_response_parts[2].strip()
        score_text = annotator_response_parts[3].strip().strip('"')
        try:
            score = float(score_text)
        except ValueError:
            raise ValueError(f"Malformed score '{score_text}' in annotator response: {annotator_response_text}")

        return {"strengths": strengths, "weaknesses": weaknesses, "score": score}
