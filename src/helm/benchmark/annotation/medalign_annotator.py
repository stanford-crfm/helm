import json
import re

from typing import Any, Dict, Optional, Union, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """You are a medical expert responsible for evaluating the quality of a model's response to a clinical instruction, given an associated Electronic Health Record (EHR).

Your goal is to assess how well the response interprets the EHR, follows clinical best practices, and aligns with the gold response in terms of accuracy, completeness, and clarity.

The instruction and EHR pair will be provided in these tags:
<user_request>
{{INPUT}}
</user_request>

The model's response will be provided in these tags:
<model_response>
{{ANSWER}}
</model_response>

The gold response (reference answer) will be provided in these tags:
<gold_response>
{{GOLD_RESPONSE}}
</gold_response>

Carefully analyze the <model_response> in the context of the <user_request>, comparing it to the <gold_response> when necessary.

For each of the following categories, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a brief justification for the score.

Evaluation Criteria

Information Completeness (1-5)
Does the response extract and integrate all relevant patient details from the EHR (e.g., history, medications, lab results)?
Does it provide all necessary steps for decision-making, treatment, or follow-up?

Clinical Accuracy (1-5)
Are the response's medical claims, treatments, and recommendations factually correct and consistent with evidence-based guidelines?
Does it avoid hallucinations, misinterpretations of the EHR, or inappropriate medical advice?

Clarity and Interpretability (1-5)
Is the response structured logically, written in clear medical language, and easy for a clinician to understand?
Does it avoid ambiguity or unnecessary complexity?

Alignment with Gold Response (1-5)
To what extent does the response align with the gold response in content, reasoning, and terminology?
If it differs, does it still provide an equally valid alternative?

Output Format
Generate a valid JSON object with your evaluation:

{
    "information_completeness": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "clinical_accuracy": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "clarity_and_interpretability": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "alignment_with_gold": {
        "score": 0,
        "explanation": "Explain why this score was given."
    }
}
Ensure the output is valid JSON.
Use double quotes (") for all JSON keys and string values.
When quoting text or sections within the explanation fields, use escaped double quotes (") to maintain valid JSON formatting.
Do not include any additional information in the output.
"""

EXPECTED_ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "information_completeness": {"score", "explanation"},
    "clinical_accuracy": {"score", "explanation"},
    "clarity_and_interpretability": {"score", "explanation"},
    "alignment_with_gold": {"score", "explanation"},
}


class MedalignAnnotator(Annotator):
    """The Medalign autograder."""

    name = "medalign"

    def __init__(self, auto_client: AutoClient, template_name: Optional[str] = None):
        self._auto_client = auto_client

    def _sanitize_model_response(self, model_response: str) -> str:
        json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
        if json_match:
            json_output = json_match.group(0)
        else:
            json_output = model_response
        return json_output
    
    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        prompt_template = PROMPT_TEMPLATE
        model_output_text = request_state.result.completions[0].text
        annotator_prompt = (
            prompt_template.replace("{{INPUT}}", request_state.instance.input.text)
            .replace("{{GOLD_RESPONSE}}", request_state.instance.references[0].output.text)
            .replace("{{ANSWER}}", model_output_text)
        )
        if not model_output_text.strip():
            hlog(
                "WARNING: MedalignAnnotator skipped sending requests to annotator models "
                "because the model response was empty"
            )
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
            "gpt": AnnotatorModelInfo(
                model_name="openai/gpt-4o-2024-05-13",
                model_deployment="stanfordhealthcare/gpt-4o-2024-05-13",
            ),
            "llama": AnnotatorModelInfo(
                model_name="meta/llama-3.3-70b-instruct",
                model_deployment="stanfordhealthcare/llama-3.3-70b-instruct",
            ),
            "claude": AnnotatorModelInfo(
                model_name="anthropic/claude-3-7-sonnet-20250219",
                model_deployment="stanfordhealthcare/claude-3-7-sonnet-20250219",
            ),
            # "gemini": AnnotatorModelInfo(
            #     model_name="google/gemini-2.0-flash-001",
            #     model_deployment="stanfordhealthcare/gemini-2.0-flash-001",
            # ),
        }
        annotations: Dict[str, Union[Optional[str], Optional[bool]]] = {"prompt_text": annotator_prompt}
        failed_counts: Dict[str, int] = {
            "gpt": 0,
            "llama": 0,
            "claude": 0,
            # "gemini": 0
        }
        for annotator_name, annotator_model_info in SHORT_NAME_TO_MODEL_INFO.items():
            failed = False
            annotator_criteria: Dict[str, Any] = {}
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
                    "WARNING: MedalignAnnotator got an error response from "
                    f"{annotator_model_info.model_name}: {annotator_response.error}. Model output: {annotator_response}"
                )
                failed = True
            else:
                try:
                    annotator_output = annotator_response.completions[0].text
                    annotator_output = self._sanitize_model_response(annotator_output)
                    annotator_criteria = json.loads(annotator_output)
                except Exception as e:
                    hlog(
                        "WARNING: MedalignAnnotator got an error parsing the response from "
                        f"{annotator_model_info.model_name}: {e}. Model output: {annotator_output}"
                    )
                    failed = True
                
                for key, value in EXPECTED_ANNOTATION_CRITERIA.items():
                    if key not in annotator_criteria:
                        hlog(
                            f"WARNING: MedalignAnnotator did not find the expected key "
                            f"'{key}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                        )
                        failed = True
                    else:
                        for subkey in value:
                            if subkey not in annotator_criteria[key]:
                                hlog(
                                    f"WARNING: MedalignAnnotator did not find the expected subkey "
                                    f"'{subkey}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                                )
                                failed = True
            failed_counts[annotator_name] += failed

            annotations[annotator_name] = annotator_criteria
        hlog(f"Failed model annotations: {failed_counts}")
        return annotations
