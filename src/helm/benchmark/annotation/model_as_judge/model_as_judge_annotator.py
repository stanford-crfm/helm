from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.src.helm.benchmark.annotation.annotator import Annotator
from helm.src.helm.clients.auto_client import AutoClient
from helm.src.helm.common.request import Request


class ModelAsJudgeAnnotator(Annotator, ABC):
    """Annotator that uses a model to generate annotations."""

    name: str = "model_as_judge"

    def __init__(self, auto_client: AutoClient, model_name: str, model_deployment: str):
        self.auto_client = auto_client
        self.model_name = model_name
        self.model_deployment = model_deployment

    def generate_annotation(self, prompt: str) -> str:
        """Generates an annotation from the model specified in AutoClient using it."""
        # TODO: Check that this method does the right thing
        annotation_request: Request = {
            "model_deployment": self.model_deployment,
            "model_name": self.model_name,
            "prompt": prompt,
        }
        request_result = self.auto_client.make_request(annotation_request)
        if request_result.success:
            return request_result.completions[0].text
        else:
            return "Error generating annotation"

    def annotate(self, request_state: RequestState, prompt: str) -> List[Dict[str, Any]]:
        """Fills the annotations field of the request state using the supplied model."""

        assert request_state.result is not None, "Annotator can only be used after the request has been processed."
        annotations: List[Dict[str, Any]] = []

        # TODO check that this is right
        annotation = self.generate_annotation(prompt)
        if annotation != "Error generating annotation":
            annotations.append({request_state.request.prompt: annotation})

        return annotations
