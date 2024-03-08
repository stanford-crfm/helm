from typing import Dict, List
from abc import abstractmethod, ABC
from dataclasses import dataclass

from helm.benchmark.annotation.annotation import Annotation
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.object_spec import ObjectSpec


class Annotator(ABC):
    """Annotator is an abstract class for annotating a request state. Annotators are used to add additional
    information to a request state that is needed for a metric to understand the request. This could be
    parsing, rendering an image based on the text completion, etc."""

    name: str
    """Name of the annotator. Should be filled in by the subclass."""

    @abstractmethod
    def annotate(self, request_state: RequestState) -> List[Dict[str, Annotation]]:
        """Fills the annotations field of the request state with additional information
        that are implementation specific."""
        pass


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

    def annotate(self, request_state: RequestState) -> List[Dict[str, Annotation]]:
        if request_state.result is None:
            raise ValueError("Annotation requires a result")
        annotations: List[Dict[str, Annotation]] = []
        for completion in request_state.result.completions:
            annotations.append({"all_caps": Annotation(completion.text.upper())})
        return annotations
