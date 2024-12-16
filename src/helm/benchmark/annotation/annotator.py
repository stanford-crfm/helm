from typing import Dict, List, Any
from abc import abstractmethod, ABC
from dataclasses import dataclass

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.object_spec import ObjectSpec


class Annotator(ABC):
    """Annotator is an abstract class for annotating a request state. Annotators are used to add additional
    information to a request state that is needed for a metric to understand the request. This could be
    parsing, rendering an image based on the text completion, etc."""

    name: str
    """Name of the annotator. Should be filled in by the subclass."""

    @abstractmethod
    def annotate(self, request_state: RequestState) -> Any:
        """Fills the annotations field of the request state with additional information
        that are implementation specific."""
        pass

    def annotate_all(self, request_states: List[RequestState]) -> List[Dict[str, Any]]:
        """Fills the annotations field of all request states with additional information
        that are implementation specific."""
        return [self.annotate(request_state) for request_state in request_states]


@dataclass(frozen=True)
class AnnotatorSpec(ObjectSpec):
    """Specifies how to create an `Annotator`.
    The user should only specify the class name.
    The arguments will be filled in by the `AnnotatorFactory`.
    """

    pass


class DummyAnnotator(Annotator):
    """A dummy annotator that does nothing."""

    name = "dummy"

    def annotate(self, request_state: RequestState) -> List[Dict[str, Any]]:
        if request_state.result is None:
            raise ValueError("Annotation requires a result")
        annotation_values: List[str] = [completion.text.upper() for completion in request_state.result.completions]
        return [{"all_caps": value} for value in annotation_values]
