from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert responsible for evaluating a hospital document.
The task requires generating either discharge instructions or a brief hospital course based
on the provided discharge summary and radiology report.

Your goal is to assess whether the generated text is clinically accurate, complete, and clear
for the intended document type. The evaluation should ensure the document aligns with the
gold response in terms of accuracy, completeness, and clarity.

The target task of either generating a discharge instruction or brief hospital course along with
the patient discharge text and radiology report will be provided in these tags:
<patient_information>
{{QUESTION}}
</patient_information>


The document will be provided in these tags:
<response>
{{RESPONSE}}
</response>

The gold standard target document (either discharge instructions or a brief hospital course)
will be provided in these tags:
<gold_response>
{{GOLD_RESPONSE}}
</gold_response>

Carefully analyze the <response> based on the <patient_information> and compare
it to the <gold_response> when necessary.

For each of the following categories, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent)
and provide a brief justification for the score.

Evaluation Criteria:
Accuracy (1-5)
- Does the document provide correct medical information based on the patient's condition and source materials?

Completeness (1-5)
- Does the document include all important information needed for the specific document type?

Clarity (1-5)
- Is the document easy to understand for the right audience — patients for discharge
  instructions or clinicians for the hospital course?

Output Format:
Generate a valid JSON object with the following structure:
{
    "accuracy": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "completeness": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "clarity": {
        "score": 0,
        "explanation": "Explain why this score was given."
    }
}

Ensure the output is valid JSON:
- Use **double quotes** (") for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes (\") to
  maintain valid JSON formatting.
- Do not include any additional information in the output.
"""

ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "accuracy": {"score", "explanation"},
    "completeness": {"score", "explanation"},
    "clarity": {"score", "explanation"},
}

ANNOTATOR_MODELS: Dict[str, AnnotatorModelInfo] = {
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
}


class DischargeMeAnnotator(LLMAsJuryAnnotator):
    """The DischargeMe autograder."""

    name = "dischargeme"

    def __init__(self, auto_client: AutoClient, template_name: Optional[str] = None):
        super().__init__(
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=ANNOTATOR_MODELS,
        )
