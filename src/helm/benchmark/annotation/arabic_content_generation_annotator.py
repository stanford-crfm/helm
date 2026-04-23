import re
from typing import Any, Dict, Union

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException


PROMPT_TEMPLATE = """Here is a user's request to generate a document, followed by the generated output. Grade the generated article based on the following rubric, and give it an integer score between 1 and 5 inclusive, where 1 is the worst and 5 is the best.

<rubric>
{{RUBRIC}}
</rubric>

<user-request>
{{USER_REQUEST}}
</user-request>

<model-output>
{{MODEL_OUTPUT}}
</model-output>

Respond with the following format:

<reasoning>
INSERT_YOUR_REASONING_HERE
</reasoning>
<score>
INSERT_YOUR_SCORE_HERE
</score>

Please output your one-sentence concise reasoning within the "reasoning" tags and your score within the "score" tags. Your reasoning should be less than 20 tokens. The score should be an integer between 1 and 5 inclusive. Do not output any other text.
"""  # noqa: E501

_RUBRICS = {
    "faithfulness": """Did the model say anything that isn’t supported by the facts in the user-provided content?

Score 1: The model frequently introduces multiple claims, details, or conclusions that are clearly not supported by or directly contradict the facts in the user-provided content.
Score 2: The model includes several unsupported statements or minor factual additions that go beyond the user-provided content, though the core response still loosely aligns with the given information.
Score 3: The model mostly relies on the user-provided content but adds at least one noticeable unsupported claim or speculative detail that is not grounded in the provided facts.
Score 4: The model stays almost entirely within the bounds of the user-provided content, with only a very minor or ambiguous addition that is not explicitly supported but does not materially affect accuracy.
Score 5: The model strictly adheres to the user-provided content and makes no claims, inferences, or details that are not directly supported by the given facts.""",
    "completeness": """Did the model include all important facts from the user-provided content?

Score 1: The model omits most of the important facts from the user-provided content, including central ideas or key details necessary for understanding.
Score 2: The model includes a few important facts but misses many key details or central points from the user-provided content.
Score 3: The model includes several important facts from the user-provided content but omits at least one significant detail or key element.
Score 4: The model includes nearly all important facts from the user-provided content, with only minor or non-essential details missing.
Score 5: The model includes all important facts from the user-provided content with no meaningful omissions.""",
    "style": """Did the model output follow the requested style, including the specified tone, intent, level of formality, register, point of view, and voice?

Score 1: The output clearly ignores or contradicts the requested style, failing to match the specified tone, intent, formality, register, point of view, or voice.
Score 2: The output shows minimal alignment with the requested style, with major inconsistencies or frequent deviations from the specified tone, formality, point of view, or voice.
Score 3: The output generally reflects the requested style but includes noticeable inconsistencies or partial mismatches in tone, formality, register, point of view, or voice.
Score 4: The output closely follows the requested style with only minor or occasional deviations in tone, intent, formality, register, point of view, or voice.
Score 5: The output fully and consistently matches the requested style, accurately maintaining the specified tone, intent, level of formality, register, point of view, and voice throughout.""",
}


class AnnotatorResponseParseFailure(NonRetriableException):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


class ArabicContentGenerationAnnotator(Annotator):
    name = "arabic_content_generation"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result

        user_request_text = request_state.request.prompt

        assert len(request_state.result.completions) == 1
        model_output_text = request_state.result.completions[0].text

        annotations: Dict[str, Dict[str, Union[int, str]]] = {}
        for rubric_id, rubric in _RUBRICS.items():
            annotator_prompt = (
                PROMPT_TEMPLATE.strip()
                .replace("{{RUBRIC}}", rubric.strip())
                .replace("{{USER_REQUEST}}", user_request_text.strip())
                .replace("{{MODEL_OUTPUT}}", model_output_text.strip())
            )

            annotator_request = Request(
                model="openai/gpt-5.4-2026-03-05",
                model_deployment="openai/gpt-5.4-2026-03-05",
                prompt=annotator_prompt,
                temperature=0.0,
                max_tokens=100,
            )
            annotator_response = self._auto_client.make_request(annotator_request)
            assert len(annotator_response.completions) == 1
            annotator_response_text = annotator_response.completions[0].text

            annotations[rubric_id] = self._parse_annotator_response(annotator_response_text)
        return annotations

    def _parse_annotator_response(self, annotator_response_text: str) -> Dict[str, Union[int, str]]:
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
            score = int(score_match.group(1).strip())
        except ValueError:
            raise AnnotatorResponseParseFailure(
                message=f"Could not parse score as float from raw request: '{annotator_response_text}'",
                response_text=annotator_response_text,
            )

        return {"reasoning": reasoning, "score": score}
