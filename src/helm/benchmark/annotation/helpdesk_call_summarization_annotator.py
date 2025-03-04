import re
from typing import Any, Dict, Optional, Union

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException


class AnnotatorResponseParseFailure(NonRetriableException):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


class HelpdeskCallSummarizationAnnotator(Annotator):
    """The Helpdesk Call Summarization autograder."""

    name = "helpdesk_call_center_summarization"

    SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
        "gpt": AnnotatorModelInfo(
            model_name="openai/gpt-4o-2024-05-13",
            model_deployment="openai/gpt-4o-2024-05-13",
        ),
        "llama": AnnotatorModelInfo(
            model_name="meta/llama-3.1-405b-instruct-turbo",
            model_deployment="together/llama-3.1-405b-instruct-turbo",
        ),
        "claude": AnnotatorModelInfo(
            model_name="anthropic/claude-3-5-sonnet-20241022",
            model_deployment="anthropic/claude-3-5-sonnet-20241022",
        ),
    }

    # Template loosely based on these references:
    # - https://ieeexplore.ieee.org/abstract/document/9946852
    # - https://arxiv.org/abs/2409.02413v1
    PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to evaluate the quality of a model-generated summary of a helpdesk call transcript.

The helpdesk call transcript and summary are provided below, delineated with start and end tags:

<call_transcript>
{{QUESTION}}
</call_transcript>
<summary>
{{PRED}}
</summary>

Evaluate the summary based on the following criteria:

- Conciseness: A high-quality summary should effectively convey the most important information from the call transcript while keeping the length brief.
- Relevance: The information presented in the summary should be relevant to the main topic.
- Coherence: A good summary should have a clear structure and flow of ideas that make it easy to understand and follow.
- Accuracy: The summary's information should be factually correct and should not contain false or misleading information.
- Completeness: The summary should a fair description of the main problems (concerns) and resolutions based on the original call transcripts.

Think step by step, then score the summary. Your reasoning should be less than 200 words. The score should be a single number between 1 to 10 inclusive.

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
        annotations: Dict[str, Union[Optional[str], Optional[float]]] = {"prompt_text": annotator_prompt}
        for annotator_name, annotator_model_info in self.SHORT_NAME_TO_MODEL_INFO.items():
            annotator_request = Request(
                model=annotator_model_info.model_name,
                model_deployment=annotator_model_info.model_deployment,
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
            reasoning: Optional[str] = None
            score: Optional[float] = None
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                hlog(
                    "WARNING: HelpdeskCallSummarizationAnnotator could not get Reasoning from annotation from "
                    f"{annotator_model_info.model_name}: {annotator_response_text}"
                )

            if score_match:
                try:
                    score = float(score_match.group(1).strip())
                except ValueError:
                    hlog(
                        "WARNING: HelpdeskCallSummarizationAnnotator could not parse Score from annotation from "
                        f"{annotator_model_info.model_name}: {annotator_response_text}"
                    )
            else:
                hlog(
                    "WARNING: HelpdeskCallSummarizationAnnotator could not get Score from annotation from "
                    f"{annotator_model_info.model_name}: {annotator_response_text}"
                )

            annotations[f"{annotator_name}_reasoning"] = reasoning
            annotations[f"{annotator_name}_score"] = score
        return annotations
