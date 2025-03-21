import json
import re

from typing import Any, Dict, Optional, Union, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of a response to a consumer health question. Your goal is to assess how well the model-generated response answers the medical question and compare it to a reference answer (gold standard).

The question will be provided in these tags:
<question>
{{QUESTION}}
</question>

The model's generated response will be provided in these tags:
<model_response>
{{MODEL_OUTPUT}}
</model_response>

The reference answer will be provided in these tags:
<gold_response>
{{REFERENCE_ANSWER}}
</gold_response>

Carefully analyze the <model_response> compared to the <gold_response> and the original <question>. For each of the following criteria, rate the model's response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:

Correctness (1-5):
- Is the medical information presented factually accurate?
- Are there any errors, hallucinations, or misinterpretations of medical concepts?

Completeness (1-5):
- Does the response cover all important medical information found in the reference answer?
- Are all key details about diagnosis, symptoms, treatments, and management captured?

Helpfulness (1-5):
- Does the response directly address the user's question and concerns?
- Is the advice practical and actionable for someone seeking medical information?

Clarity (1-5):
- Is the response presented in clear, understandable language appropriate for a non-medical audience?
- Is medical terminology explained when necessary?

Alignment with Reference (1-5):
- To what extent does the model's response align with the reference answer in terms of content, approach, and recommendations?
- Are there significant discrepancies between the model's response and the reference?

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:

{
    "correctness": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "completeness": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "helpfulness": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "clarity": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "alignment_with_reference": {
        "score": 1,
        "explanation": "Explain why this score was given."
    }
}
Ensure the output is valid JSON.
Use single quotes (') inside the explanation fields when quoting text or sections to avoid invalid JSON formatting.
Do not include any additional information in the output.
"""

EXPECTED_ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "correctness": {"score", "explanation"},
    "completeness": {"score", "explanation"},
    "helpfulness": {"score", "explanation"},
    "clarity": {"score", "explanation"},
    "alignment_with_reference": {"score", "explanation"},
}


class MediQAAnnotator(Annotator):
    """The MediQA autograder."""

    name = "medi_qa"

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
            prompt_template.replace("{{QUESTION}}", request_state.instance.input.text)
            .replace("{{REFERENCE_ANSWER}}", request_state.instance.references[0].output.text)
            .replace("{{MODEL_OUTPUT}}", model_output_text)
        )
        if not model_output_text.strip():
            hlog(
                "WARNING: MediQAAnnotator skipped sending requests to annotator models "
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
                    "WARNING: MediQAAnnotator got an error response from "
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
                        "WARNING: MediQAAnnotator got an error parsing the response from "
                        f"{annotator_model_info.model_name}: {e}. Model output: {annotator_output}"
                    )
                    failed = True
                
                for key, value in EXPECTED_ANNOTATION_CRITERIA.items():
                    if key not in annotator_criteria:
                        hlog(
                            f"WARNING: MediQAAnnotator did not find the expected key "
                            f"'{key}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                        )
                        failed = True
                    else:
                        for subkey in value:
                            if subkey not in annotator_criteria[key]:
                                hlog(
                                    f"WARNING: MediQAAnnotator did not find the expected subkey "
                                    f"'{subkey}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                                )
                                failed = True
            failed_counts[annotator_name] += failed

            annotations[annotator_name] = annotator_criteria
        hlog(f"Failed model annotations: {failed_counts}")
        return annotations
