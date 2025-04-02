from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert responsible for evaluating the quality of a
response to a clinical instruction, given an associated Electronic Health Record (EHR).

Your goal is to assess how well the response interprets the EHR, follows clinical best practices,
and aligns with the gold response in terms of accuracy, completeness, and clarity.

The instruction and EHR pair will be provided in these tags:
<user_request>
{{QUESTION}}
</user_request>

The response will be provided in these tags:
<response>
{{RESPONSE}}
</response>

The gold response (reference answer) will be provided in these tags:
<gold_response>
{{GOLD_RESPONSE}}
</gold_response>

Carefully review the <response> based on the <user_request> and compare it to
the <gold_response> when needed.
For each category below, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent)
and provide a brief explanation for your score:

Evaluation Criteria:
Accuracy (1-5)
- Does the response provide correct medical information based on the EHR and clinical guidelines?

Completeness (1-5)
- Does the response include all important patient information and address the clinical question?

Clarity (1-5)
- Is the response written clearly so clinicians can easily understand it?

Output Format:
Generate a valid JSON object with your evaluation:
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


class MedalignAnnotator(LLMAsJuryAnnotator):
    """The Medalign autograder."""

    name = "medalign"

    def __init__(self, auto_client: AutoClient, template_name: Optional[str] = None):
        super().__init__(
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=ANNOTATOR_MODELS,
        )
