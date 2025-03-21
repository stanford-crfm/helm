import json
import re

from typing import Any, Dict, Optional, Union, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of a generated clinical note based on a doctor-patient conversation.
Your goal is to assess how well the model-generated note captures the clinical information from the conversation and compare it to a reference note (gold standard).
The conversation will be provided in these tags: 
<conversation> 
{{CONVERSATION}} 
</conversation>
The model's generated note will be provided in these tags: 
<model_response> 
{{MODEL_OUTPUT}} 
</model_response>
The reference note will be provided in these tags: 
<gold_response> 
{{REFERENCE_NOTE}} 
</gold_response>
Carefully analyze the <model_response> compared to the <gold_response> and the original <conversation>. For each of the following criteria, rate the model's response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria
Completeness (1-5): 
- Does the model's note include all important medical information found in the reference?
- Are all key details about patient history, symptoms, exam findings, results, and treatment plans captured?

Accuracy (1-5):
- Is the information presented factually correct compared to the reference?
- Are there any errors, hallucinations, or misinterpretations?

Clarity and Structure (1-5):
- Is the note well-organized into the four standard sections (HPI, Physical Exam, Results, Assessment & Plan)?
- Does the note follow logical clinical documentation conventions?

Alignment with Gold Response (1-5)
- To what extent does the model's note align with the reference note in terms of content, clinical approach, and terminology?
- Are there significant discrepancies between the model's note and the reference?

Output Format
Output the evaluation as a single valid JSON object matching the following structure:
{
  "completeness": {
    "score": 3,
    "explanation": "Explain why this score was given."
  },
  "accuracy": {
    "score": 4,
    "explanation": "Explain why this score was given."
  },
  "clarity_and_structure": {
    "score": 2,
    "explanation": "Explain why this score was given."
  },
  "alignment_with_gold_response": {
    "score":3,
    "explanation": "Explain why this score was given."
  }
}
Ensure the output is valid JSON.
Use single quotes (') inside the explanation fields when quoting text or sections to avoid invalid JSON formatting.
Do not include any additional information in the output.
"""

EXPECTED_ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "completeness": {"score", "explanation"},
    "accuracy": {"score", "explanation"},
    "clarity_and_structure": {"score", "explanation"},
    "alignment_with_gold_response": {"score", "explanation"},
}


class ACIBenchAnnotator(Annotator):
    """The ACIBench autograder."""

    name = "aci_bench"

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
            prompt_template.replace("{{CONVERSATION}}", request_state.instance.input.text)
            .replace("{{REFERENCE_NOTE}}", request_state.instance.references[0].output.text)
            .replace("{{MODEL_OUTPUT}}", model_output_text)
        )
        if not model_output_text.strip():
            hlog(
                "WARNING: ACIBenchAnnotator skipped sending requests to annotator models "
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
                    "WARNING: ACIBenchAnnotator got an error response from "
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
                        "WARNING: ACIBenchAnnotator got an error parsing the response from "
                        f"{annotator_model_info.model_name}: {e}. Model output: {annotator_output}"
                    )
                    failed = True
                
                for key, value in EXPECTED_ANNOTATION_CRITERIA.items():
                    if key not in annotator_criteria:
                        hlog(
                            f"WARNING: ACIBenchAnnotator did not find the expected key "
                            f"'{key}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                        )
                        failed = True
                    else:
                        for subkey in value:
                            if subkey not in annotator_criteria[key]:
                                hlog(
                                    f"WARNING: ACIBenchAnnotator did not find the expected subkey "
                                    f"'{subkey}' in the response from {annotator_model_info.model_name}. Model output: {annotator_output}"
                                )
                                failed = True
            failed_counts[annotator_name] += failed

            annotations[annotator_name] = annotator_criteria
        hlog(f"Failed model annotations: {failed_counts}")
        return annotations
