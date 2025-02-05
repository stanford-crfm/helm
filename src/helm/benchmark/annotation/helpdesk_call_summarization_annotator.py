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


class HelpdeskCallSummarizationAnnotator(Annotator):
    """The Helpdesk Call Summarization autograder."""

    name = "helpdesk_call_center_summarization"

    PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to evaluate the quality of a model-generated summary of a helpdesk call transcript.

The helpdesk call transcript and summary are provided below, delineated with start and end tags:

<call_transcript>
{{QUESTION}}
</call_transcript>
<summary>
{{PRED}}
</summary>

Evaluate the summary based on the following criteria:

- Conciseness: A high-quality summary should effectively convey the most important information from the original source while keeping the length brief.
- Relevance: The information presented in the summary should be relevant to the main topic.
- Coherence: A good summary should have a clear structure and flow of ideas that make it easy to understand and follow.
- Accuracy: The summary's information should be factually correct and should not contain false or misleading information.

Think step by step, then score the summary. Your reasoning should be less than 100 words. The score should be a single number between 1 to 10 inclusive.

Please respond with your output and reasoning in the following format, your reasoning within <reasoning></reasoning> tags and your score within <score></score> tags, without any other output:

<reasoning>INSERT_YOUR_REASONING_HERE</reasoning>
<score>INSERT_YOUR_SCORE_HERE</score>
"""  # noqa: E501

    PATTERN = r"^\s*reason:(.*)##(.*)"

    def __init__(self, auto_client: AutoClient):
        self._auto_client = auto_client

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        prediction_text = request_state.result.completions[0].text

        question_text = request_state.instance.input.text

        annotator_prompt = self.PROMPT_TEMPLATE.replace("{{QUESTION}}", question_text).replace(
            "{{PRED}}", prediction_text
        )
        annotator_request = Request(
            model="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
            prompt=annotator_prompt,
            temperature=0.0,
            max_tokens=512,
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
