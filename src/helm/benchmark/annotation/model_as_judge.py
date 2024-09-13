import re
from typing import Dict

from helm.clients.auto_client import AutoClient
from helm.common.request import Request


def score_with_reasoning(
    auto_client: AutoClient,
    annotator_prompt: str,
    annotator_model: str,
    annotator_model_deployment: str,
) -> Dict:
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
    reasoning_match = re.search(r"<reasoning>(.*?)</?\s*reasoning>", annotator_response_text, re.DOTALL | re.IGNORECASE)
    score_match = re.search(r"<score>(.*?)</?\s*score>", annotator_response_text, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Reasoning not found"
    score = score_match.group(1).strip() if score_match else None

    if isinstance(score, (float, int, str)):
        try:
            score = float(score)
        except ValueError:
            raise Exception(f"Failed to parse score in: {annotator_response_text}")
    else:
        raise Exception(f"Score not found in: {annotator_response_text}")

    return {"reasoning": reasoning, "score": score}
