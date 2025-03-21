import json
import re

from typing import Any, Dict, Optional, Union, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """You are a mental health expert tasked with evaluating the quality of a counselor response in a therapy conversation. Your goal is to assess how well the model-generated response demonstrates therapeutic skills compared to a reference response (gold standard).

The conversation history and question will be provided in these tags:
<conversation>
{{CONVERSATION_HISTORY}}
</conversation>

The model's generated counselor response will be provided in these tags:
<model_response>
{{MODEL_OUTPUT}}
</model_response>

The reference counselor response will be provided in these tags:
<gold_response>
{{REFERENCE_RESPONSE}}
</gold_response>

Carefully analyze the <model_response> compared to the <gold_response> and the original <conversation>. For each of the following criteria, rate the model's response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:

Conversational Leadership (1-5):
- Does the counselor effectively move the conversation in a productive direction?
- Does the counselor follow therapeutic principles, allowing appropriate space for the client while guiding towards resolution?
- Does the counselor address the client's presenting issues and concerns?

Counseling Micro Skills (1-5):
- Does the counselor demonstrate appropriate micro skills such as paraphrasing, active listening, validation, and positive reinforcement?
- Does the counselor avoid challenging or commanding the client inappropriately?
- Are skills applied consistently and skillfully to support the client?

Contextual Understanding (1-5):
- Does the counselor demonstrate understanding of the client's narrative and prior context?
- Does the counselor incorporate important details from the conversation history?
- Does the counselor attend to safety concerns and the client's present state appropriately?

Therapeutic Communication (1-5):
- Is the counselor's style, tone, and language choice appropriate, empathetic, and clear?
- Is the communication tailored to the client's specific needs and situation?
- Does the language avoid potentially triggering or confusing elements?

Alignment with Reference (1-5):
- To what extent does the model's response align with the reference response in approach and content?
- Does it address the same key concerns highlighted in the reference response?
- Does it demonstrate similar therapeutic quality and effectiveness?

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:

{
    "conversational_leadership": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "counseling_micro_skills": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "contextual_understanding": {
        "score": 1,
        "explanation": "Explain why this score was given."
    },
    "therapeutic_communication": {
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
    "conversational_leadership": {"score", "explanation"},
    "counseling_micro_skills": {"score", "explanation"},
    "contextual_understanding": {"score", "explanation"},
    "therapeutic_communication": {"score", "explanation"},
    "alignment_with_reference": {"score", "explanation"},
}


class MentalHealthAnnotator(Annotator):
    """The MentalHealth autograder."""

    name = "mental_health"

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
            prompt_template.replace("{{CONVERSATION_HISTORY}}", request_state.instance.input.text)
            .replace("{{REFERENCE_RESPONSE}}", request_state.instance.references[0].output.text)
            .replace("{{MODEL_OUTPUT}}", model_output_text)
        )
        if not model_output_text.strip():
            hlog(
                "WARNING: MentalHealthAnnotator skipped sending requests to annotator models "
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
                    "WARNING: MentalHealthAnnotator got an error response from "
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
                        "WARNING: MentalHealthAnnotator got an error parsing the response from "
                        f"{annotator_model_info.model_name}: {e}. Model output: {annotator_output}"
                    )
                    failed = True
                
                for key, value in EXPECTED_ANNOTATION_CRITERIA.items():
                    if key not in annotator_criteria:
                        hlog(
                            f"WARNING: MentalHealthAnnotator did not find the expected key "
                            f"'{key}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                        )
                        failed = True
                    else:
                        for subkey in value:
                            if subkey not in annotator_criteria[key]:
                                hlog(
                                    f"WARNING: MentalHealthAnnotator did not find the expected subkey "
                                    f"'{subkey}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                                )
                                failed = True
            failed_counts[annotator_name] += failed

            annotations[annotator_name] = annotator_criteria
        hlog(f"Failed model annotations: {failed_counts}")
        return annotations
