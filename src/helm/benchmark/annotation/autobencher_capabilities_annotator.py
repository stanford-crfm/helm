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


class AutoBencherCapabilitiesAnnotator(Annotator):
    """The AutoBencher Capabilities autograder."""

    name = "autobencher_capabilities"

    PROMPT_TEMPLATE = """Here is a question, a gold answer, and a response from a model. Compare the model's response to the gold answer and judge whether the model's response is correct or not.

If the model's response is correct, score it as 1. If the model's response is incorrect, score it as 0.

We'd still consider the prediction to be correct if
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

The user's question, the golden answer, and model's response are provided below, delineated with start and end tags:

<question>
{{QUESTION}}
</question>
<gold_answer>
{{GOLD}}
</gold_answer>
<model_response>
{{PRED}}
</model_response>

Please output your one-sentence concise reasoning within the "reasoning" tags and your score within the "score" tags.
Your reasoning should be less than 20 tokens. The score should be a single number with no other output.
Only output a tag-delimited object with the following format:

<reasoning>
INSERT_YOUR_REASONING_HERE
</reasoning>
<score>
INSERT_YOUR_SCORE_HERE
</score>"""  # noqa: E501

    PATTERN = r"^\s*reason:(.*)##(.*)"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        prediction_text = request_state.result.completions[0].text

        question_text = request_state.instance.input.text
        correct_references = request_state.instance.all_correct_references
        assert len(correct_references) == 1
        gold_text = correct_references[0].output.text

        annotator_prompt = (
            self.PROMPT_TEMPLATE.replace("{{QUESTION}}", question_text)
            .replace("{{PRED}}", prediction_text)
            .replace("{{GOLD}}", gold_text)
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
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
