from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Reference
from helm.common.request import RequestResult


def gather_generated_image_locations(request_result: RequestResult) -> List[str]:
    """Gathers the locations (file paths or URLs) of the generated images."""
    image_locations: List[str] = []
    for image in request_result.completions:
        # Models like DALL-E 2 can skip generating images for prompts that violate their content policy
        if image.file_location is None:
            return []

        image_locations.append(image.file_location)
    return image_locations


def get_gold_image_location(request_state: RequestState) -> str:
    """Returns the first gold image location."""
    references: List[Reference] = request_state.instance.references
    assert len(references) > 0 and references[0].output.file_path is not None, "Expected at least one gold image"
    return references[0].output.file_path
