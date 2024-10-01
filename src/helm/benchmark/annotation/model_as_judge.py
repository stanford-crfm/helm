from dataclasses import dataclass
import re
from typing import Dict, Optional, TypedDict, Union

from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class AnnotatorResponseParseFailure(Exception):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


@dataclass
class _AnnotatorModelInfo:
    model_name: str
    model_deployment: str


def score_with_reasoning_with_gpt_and_llama(
    auto_client: AutoClient,
    annotator_prompt: str,
) -> Dict[str, Optional[Union[str, float]]]:
    """EXPERIMENTAL: DO NOT USE IN PRODUCTION

    Score using GPT-4o and Llama 3.1 for safety scenarios in HELM Safety."""
    # TODO: Make this configurable
    SHORT_NAME_TO_MODEL_INFO: Dict[str, _AnnotatorModelInfo] = {
        "gpt": _AnnotatorModelInfo(model_name="openai/gpt-4o-2024-05-13", model_deployment="openai/gpt-4o-2024-05-13"),
        "llama": _AnnotatorModelInfo(
            model_name="meta/llama-3.1-405b-instruct-turbo", model_deployment="together/llama-3.1-405b-instruct-turbo"
        ),
    }
    result: Dict[str, Optional[Union[str, float]]] = {"prompt_text": annotator_prompt}
    for short_name, model_info in SHORT_NAME_TO_MODEL_INFO.items():
        try:
            score_and_reasoning = score_with_reasoning(
                auto_client,
                annotator_prompt,
                annotator_model=model_info.model_name,
                annotator_model_deployment=model_info.model_deployment,
            )
            result[f"{short_name}_score"] = score_and_reasoning["score"]
            result[f"{short_name}_reasoning"] = score_and_reasoning["reasoning"]
        except AnnotatorResponseParseFailure as e:
            result[f"{short_name}_score"] = None
            result[f"{short_name}_reasoning"] = None
            result[f"{short_name}_response_text"] = e.response_text
    return result


class ScoreAndReasoning(TypedDict):
    score: float
    reasoning: str


def score_with_reasoning(
    auto_client: AutoClient,
    annotator_prompt: str,
    annotator_model: str,
    annotator_model_deployment: str,
) -> ScoreAndReasoning:
    annotator_request = Request(
        model=annotator_model,
        model_deployment=annotator_model_deployment,
        prompt=annotator_prompt,
        temperature=0.0,
        max_tokens=256,
    )
    annotator_response = auto_client.make_request(annotator_request)
    if not annotator_response.success:
        raise Exception(f"Annotation request failed: {annotator_response.error}")
    assert len(annotator_response.completions) == 1
    annotator_response_text = annotator_response.completions[0].text

    # fuzzy match regex check, allows for different casing, or forgetting / in end tag
    reasoning_match = re.search(
        r"<\s*reasoning\s*>(.*?)<\/?\s*reasoning\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE
    )
    score_match = re.search(r"<\s*score\s*>(.*?)<\/?\s*score\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE)
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
