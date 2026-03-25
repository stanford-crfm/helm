import re
from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException


class AnnotatorResponseParseFailure(NonRetriableException):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


class ArabicFinanceCalculationAnnotator(Annotator):
    name = "arabic_finance_calculation"

    PROMPT_TEMPLATE = """Here is a math question, a reference answer, and a model output, delineated with start and end tags. Judge whether the model output is mathematically equivalent to the reference answer.

<question>
{{question}}
</question>

<reference-answer>
{{reference_answer}}
</reference-answer>

<model-output>
{{model_output}}
</model-output>

Respond with the following format:

<reasoning>
INSERT_YOUR_REASONING_HERE
</reasoning>
<score>
INSERT_YOUR_SCORE_HERE
</score>

Please output your one-sentence concise reasoning within the "reasoning" tags and your score within the "score" tags. Your reasoning should be less than 20 tokens. The score should be either 1 or 0. Do not output any other text.
"""  # noqa: E501

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        question_text = request_state.instance.input.text
        reference_text = request_state.instance.references[0].output.text
        prediction_text = request_state.result.completions[0].text

        annotator_prompt = (
            self.PROMPT_TEMPLATE.replace("{{question}}", question_text)
            .replace("{{reference_answer}}", reference_text)
            .replace("{{model_output}}", prediction_text)
        )
        annotator_request = Request(
            model="openai/gpt-5.4-2026-03-05",
            model_deployment="openai/gpt-5.4-2026-03-05",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=100,
        )
        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            raise Exception(f"Annotation request failed: {annotator_response.error}")
        assert len(annotator_response.completions) == 1
        annotator_response_text = annotator_response.completions[0].text
        # fuzzy match regex check, allows for different casing, or forgetting / in end tag
        reasoning_match = re.search(
            r"<\s*reasoning\s*>(.*?)<\/?\s*reasoning\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE
        )
        score_match = re.search(
            r"<\s*score\s*>(.*?)<\/?\s*score\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE
        )
        if not reasoning_match or not score_match:
            raise AnnotatorResponseParseFailure(
                message=f"Could not parse markup in raw response: '{annotator_response_text}'",
                response_text=annotator_response_text,
            )
        reasoning = reasoning_match.group(1).strip()
        try:
            score = float(score_match.group(1).strip())
        except ValueError:
            raise AnnotatorResponseParseFailure(
                message=f"Could not parse score as float from raw request: '{annotator_response_text}'",
                response_text=annotator_response_text,
            )

        return {"reasoning": reasoning, "score": score}
