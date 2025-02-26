from typing import Any, Dict, Optional, Union
from importlib.resources import files

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request


# Following https://github.com/KbsdJames/Omni-MATH/blob/23be225c8e268df51990f6c5c1448f34d3b56911/GPT_eval/get_result.py
def _parse_report(report):
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

    def __init__(self, auto_client: AutoClient, template_name: Optional[str] = None):
        self._auto_client = auto_client
        self._template_name = template_name or "gpt_evaluation_zero_shot_template"
        template_path = files("helm.benchmark.annotation.omni_math").joinpath(f"{self._template_name}.txt")
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
            hlog(
                "WARNING: OmniMATHAnnotator skipped sending requests to annotator models "
                "because the model response was empty"
            )
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
            "gpt": AnnotatorModelInfo(
                model_name="openai/gpt-4o-2024-05-13",
                model_deployment="openai/gpt-4o-2024-05-13",
            ),
            "llama": AnnotatorModelInfo(
                model_name="meta/llama-3.1-405b-instruct-turbo",
                model_deployment="together/llama-3.1-405b-instruct-turbo",
            ),
            "claude": AnnotatorModelInfo(
                model_name="anthropic/claude-3-5-sonnet-20241022",
                model_deployment="anthropic/claude-3-5-sonnet-20241022",
            ),
        }
        annotations: Dict[str, Union[Optional[str], Optional[bool]]] = {"prompt_text": annotator_prompt}

        for annotator_name, annotator_model_info in SHORT_NAME_TO_MODEL_INFO.items():
            student_final_answer: Optional[str] = None
            equivalence_judgement: Optional[bool] = None
            justification: Optional[str] = None
            annotator_request = Request(
                model=annotator_model_info.model_name,
                model_deployment=annotator_model_info.model_deployment,
                prompt=annotator_prompt,
                temperature=0.0,
                max_tokens=4096,
            )
            annotator_response = self._auto_client.make_request(annotator_request)
            if not annotator_response.success:
                hlog(
                    "WARNING: OmniMATHAnnotator got an error response from "
                    f"{annotator_model_info.model_name}: {annotator_response.error}"
                )
            else:
                assert len(annotator_response.completions) == 1
                annotator_response_text = annotator_response.completions[0].text
                report_parts: Dict[str, str] = _parse_report(annotator_response_text)
                try:
                    student_final_answer = report_parts["Student Final Answer"]
                except KeyError:
                    hlog(
                        "WARNING: OmniMATHAnnotator could not get Student Final Answer from annotation from "
                        f"{annotator_model_info.model_name}: {annotator_response_text}"
                    )

                try:
                    justification = report_parts["Justification"].strip().removesuffix("=== report over ===").strip()
                except KeyError:
                    hlog(
                        "WARNING: OmniMATHAnnotator could not get Justification from annotation from "
                        f"{annotator_model_info.model_name}: {annotator_response_text}"
                    )

                try:
                    equivalence_judgement_str = report_parts["Equivalence Judgement"].strip().upper()
                    if equivalence_judgement_str == "TRUE":
                        equivalence_judgement = True
                    elif equivalence_judgement_str == "FALSE":
                        equivalence_judgement = False
                    else:
                        hlog(
                            "WARNING: OmniMATHAnnotator got a non-boolean Equivalence Judgement from annotation from "
                            f"{annotator_model_info.model_name}: {equivalence_judgement_str}"
                        )
                except KeyError:
                    hlog(
                        "WARNING: OmniMATHAnnotator could not get Equivalence Judgement from annotation from "
                        f"{annotator_model_info.model_name}: {annotator_response_text}"
                    )

            annotations[f"{annotator_name}_student_final_answer"] = student_final_answer
            annotations[f"{annotator_name}_justification"] = justification
            annotations[f"{annotator_name}_equivalence_judgement"] = equivalence_judgement
        return annotations
