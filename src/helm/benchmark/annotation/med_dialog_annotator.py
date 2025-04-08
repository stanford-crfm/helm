from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of medical dialogue summaries.

Your goal is to assess how well the summary captures the patient-doctor conversation,
and how it compares to the gold response in terms of accuracy, completeness, and clarity.

The patient-doctor conversation will be provided in these tags:
<conversation>
{{QUESTION}}
</conversation>

The response will be provided in these tags:
<response>
{{RESPONSE}}
</response>

The reference response will be provided in these tags:
<gold_response>
{{GOLD_RESPONSE}}
</gold_response>

Carefully review the <response> and compare it to the <gold_response> when needed.

Please evaluate the generated summary on a scale of 1-5 (1 = poor, 5 = excellent)
for each of these three key dimensions:

Evaluation Criteria:
Accuracy (1-5)
- Does the summary correctly capture the main medical issue and clinical details from the conversation?

Completeness (1-5)
- Does the summary include all important medical information from the conversation?

Clarity (1-5)
- Is the summary easy to understand for clinical use?

Output Format:
Output your evaluation as a single valid JSON object matching the following structure:
{
    "accuracy": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given."
    },
    "completeness": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given."
    },
    "clarity": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given."
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


class MedDialogAnnotator(LLMAsJuryAnnotator):
    """The MedDialog autograder."""

    name = "med_dialog"

    def __init__(self, auto_client: AutoClient, template_name: Optional[str] = None):
        super().__init__(
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=ANNOTATOR_MODELS,
        )
