from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of a
response to a consumer health question.
Your goal is to assess how well the generated response answers the medical question,
and how it compares to the gold response in terms of accuracy, completeness, and clarity.

The question will be provided in these tags:
<question>
{QUESTION}
</question>

The response will be provided in these tags:
<response>
{RESPONSE}
</response>

The reference answer will be provided in these tags:
<gold_response>
{GOLD_RESPONSE}
</gold_response>

Carefully analyze the <response> compared to the <gold_response> and the original <question>.
For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent),
and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Is the medical information correct?

Completeness (1-5)
- Does the response fully answer the question and include important medical details?

Clarity (1-5)
- Is the response easy for consumers to understand?

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


class MediQAAnnotator(LLMAsJuryAnnotator):
    """The MediQA autograder."""

    def __init__(
        self,
        auto_client: AutoClient,
        annotator_models: Dict[str, AnnotatorModelInfo],
        template_name: Optional[str] = None,
    ):
        super().__init__(
            name="medi_qa",
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=annotator_models,
        )
