from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


# Following https://github.com/KbsdJames/Omni-MATH/blob/main/GPT_eval/get_result.py
def parse_report(report):
    parts = report.split("## ")
    data = {}
    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        if title == "Justification":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''
    return data


class OmniMATHAnnotator(Annotator):
    """The OmniMATH autograder."""

    name = "omnimath"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        with open("src/helm/benchmark/annotation/omnimath/gpt_evaluation_template.txt") as f:
            self._score_template = f.read()

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert request_state.instance.extra_data
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            return {"prompt_text": annotator_prompt, "correctness": 0.0}
        prompt_template = self._score_template

        annotator_prompt = (
            prompt_template.replace("{{Problem}}", request_state.instance.input.text)
            .replace("{{Reference Answer}}", request_state.instance.extra_data["answer"])
            .replace("{{Solution}}", model_output_text)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=1000,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text

        info = parse_report(annotator_response_text)

        # correctness = info['Equivalence Judgement']
        correctness = info.get("Equivalence Judgement", "FALSE")
        
        if correctness == "TRUE":
            return {"prompt_text": annotator_prompt, "correctness": 1.0}
        else:
            return {"prompt_text": annotator_prompt, "correctness": 0.0}
