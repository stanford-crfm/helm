from typing import Any
from importlib.resources import files

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
            data[title] = lines[1].strip() if len(lines) > 1 else ""
    return data


class OmniMATHAnnotator(Annotator):
    """The Omni-MATH autograder."""

    name = "omni_math"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client
        template_path = files("helm.benchmark.annotation.omni_math").joinpath("gpt_evaluation_template.txt")
        with template_path.open("r") as file:
            self._score_template = file.read()

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        prompt_template = self._score_template
        model_output_text = request_state.result.completions[0].text
        annotator_prompt = (
            prompt_template.replace("{{Problem}}", request_state.instance.input.text)
            .replace("{{Reference Answer}}", request_state.instance.references[0].output.text)
            .replace("{{Solution}}", model_output_text)
        )
        if not model_output_text.strip():
            return {"prompt_text": annotator_prompt, "correctness": 0.0}

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

        equivalence_judgement = info.get("Equivalence Judgement", "")
        student_final_answer = info.get("Student Final Answer", "")
        justification = info.get("Justification", "").strip().removesuffix("=== report over ===").strip()

        return {
            "student_final_answer": student_final_answer,
            "equivalence_judgement": equivalence_judgement,
            "justification": justification,
        }
