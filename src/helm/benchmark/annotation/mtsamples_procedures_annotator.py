from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of a
generated response of a clinical scenario.
Your goal is to assess how well the response addresses the patient case, follows clinical best practices,
and compares to the gold response in terms of accuracy, completeness, and clarity.

The user's request will be provided in these tags:
<user_request>
{QUESTION}
</user_request>

The response will be provided in these tags:
<response>
{RESPONSE}
</response>

Some potential correct responses will be provided in these tags:
<gold_response>
{GOLD_RESPONSE}
</gold_response>

Carefully analyze the <response>.
For each of the following categories, rate the Response on a scale of 1 to 5 (1 = very poor, 5 = excellent),
and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Does the response provide correct medical advice based on clinical guidelines and the patient case?

Completeness (1-5)
- Does the response cover all important aspects of care for this clinical scenario?

Clarity (1-5)
- Is the response written clearly and organized in a way that clinicians can easily follow?

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:
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


class MTSamplesProceduresAnnotator(LLMAsJuryAnnotator):
    """The MTSamplesProcedures autograder."""

    def __init__(
        self,
        auto_client: AutoClient,
        annotator_models: Dict[str, AnnotatorModelInfo],
        template_name: Optional[str] = None,
    ):
        super().__init__(
            name="mtsamples_procedures",
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=annotator_models,
        )
