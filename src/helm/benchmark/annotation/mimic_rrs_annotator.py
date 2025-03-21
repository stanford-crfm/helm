import json
import re

from typing import Any, Dict, Optional, Union, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """You are tasked with evaluating the quality of the generated impression section of a radiology report based on the provided findings.
The user's request will be provided in these tags:
<user_request>
{{QUESTION}}
</user_request>
The model's response will be provided in these tags:
<model_response>
{{ANSWER}}
</model_response>
Some potential correct responses will be provided in these tags:
<gold_response>
{{GOLD_RESPONSE}}
</gold_response>
Carefully analyze the <model_response>.
For each of the following categories, rate the Response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

You are tasked with evaluating the quality of the impression section of a radiology report based on the provided findings. Your evaluation should focus on the following criteria:

Conciseness (1-5)
Is the impression section succinct and to the point?

Clarity (1-5)
Are the findings clearly articulated and easy to understand?

Relevance (1-5)
Does the impression accurately reflect the key findings from the report?

Diagnostic Insight (1-5)
Does the impression provide useful insights without making definitive diagnoses or treatment recommendations?

Output the evaluation as a single valid JSON object matching the following structure:
{
    "conciseness": {
        ""score"": 1,
        ""explanation"": ""Explain why this score was given.""  
    },
    "clarity": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "relevance": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "diagnostic_insight": {
        "score": 1,
        "explanation": "Explain why this score was given."
    }
}
Ensure the output is valid JSON.
Use single quotes (') inside the explanation fields when quoting text or sections to avoid invalid JSON formatting.
Do not include any additional information in the output.
"""

EXPECTED_ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "conciseness": {"score", "explanation"},
    "clarity": {"score", "explanation"},
    "relevance": {"score", "explanation"},
    "diagnostic_insight": {"score", "explanation"},
}


class MIMICRRSAnnotator(Annotator):
    """The MIMICRRS autograder."""

    name = "mimic_rrs"

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
            .replace("{{GOLD_RESPONSE}}", request_state.instance.references[0].output.text)
            .replace("{{ANSWER}}", model_output_text)
        )
        if not model_output_text.strip():
            hlog(
                "WARNING: MIMICRRSAnnotator skipped sending requests to annotator models "
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
                    "WARNING: MIMICRRSAnnotator got an error response from "
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
                        "WARNING: MIMICRRSAnnotator got an error parsing the response from "
                        f"{annotator_model_info.model_name}: {e}. Model output: {annotator_output}"
                    )
                    failed = True
                
                for key, value in EXPECTED_ANNOTATION_CRITERIA.items():
                    if key not in annotator_criteria:
                        hlog(
                            f"WARNING: MIMICRRSAnnotator did not find the expected key "
                            f"'{key}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                        )
                        failed = True
                    else:
                        for subkey in value:
                            if subkey not in annotator_criteria[key]:
                                hlog(
                                    f"WARNING: MIMICRRSAnnotator did not find the expected subkey "
                                    f"'{subkey}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                                )
                                failed = True
            failed_counts[annotator_name] += failed

            annotations[annotator_name] = annotator_criteria
        hlog(f"Failed model annotations: {failed_counts}")
        return annotations
