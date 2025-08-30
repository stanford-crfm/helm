from typing import Dict, Optional, Set

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.clients.auto_client import AutoClient


PROMPT_TEMPLATE = """You are a medical expert tasked with evaluating the quality of a
generated clinical note based on a doctor-patient conversation.
Your goal is to assess how well the note captures the clinical information from the conversation and
compare it to the reference note (gold standard) in terms of accruacy, completeness and clarity.
The conversation will be provided in these tags:
<conversation>
{QUESTION}
</conversation>
The generated note will be provided in these tags:
<response>
{RESPONSE}
</response>
The reference note will be provided in these tags:
<gold_response>
{GOLD_RESPONSE}
</gold_response>
Carefully review the <response> based on the <conversation> and compare it to the <gold_response> when needed.

For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent),
and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Does the note provide correct clinical information based on the conversation?

Completeness (1-5)
- Does the note include all important medical details from the conversation?

Clarity (1-5)
- Is the note written clearly and organized in a standard clinical format for clinicians

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


class ACIBenchAnnotator(LLMAsJuryAnnotator):
    """The ACIBench autograder."""

    def __init__(
        self,
        auto_client: AutoClient,
        annotator_models: Dict[str, AnnotatorModelInfo],
        template_name: Optional[str] = None,
    ):
        super().__init__(
            name="aci_bench",
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=annotator_models,
        )
