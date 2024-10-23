import base64
from typing import List, Optional

import requests

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Reference
from helm.common.request import RequestResult


def get_contents_as_base64(url: str) -> str:
    return base64.b64encode(get_contents_as_base64(url)).decode('utf-8')


def get_contents_as_bytes(url: str) -> bytes:
    # Fetch the audio file and convert it to a base64 encoded string
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def gather_generated_image_locations(request_result: RequestResult) -> List[str]:
    """Gathers the locations (file paths or URLs) of the generated images."""
    image_locations: List[str] = []
    for image in request_result.completions:
        # Models like DALL-E 2 can skip generating images for prompts that violate their content policy
        if image.multimodal_content is None or image.multimodal_content.size == 0:
            return []

        location: Optional[str] = image.multimodal_content.media_objects[0].location
        if location is not None:
            image_locations.append(location)
    return image_locations


def get_gold_image_location(request_state: RequestState) -> str:
    """Returns the first gold image location."""
    references: List[Reference] = request_state.instance.references
    assert (
        len(references) > 0
        and references[0].output.multimedia_content is not None
        and references[0].output.multimedia_content.size > 0
        and references[0].output.multimedia_content.media_objects[0].location is not None
    ), "Expected at least one gold image"
    return references[0].output.multimedia_content.media_objects[0].location
