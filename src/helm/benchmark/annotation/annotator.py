from typing import Dict, Any, List
from abc import abstractmethod, ABC
from dataclasses import dataclass

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.object_spec import ObjectSpec


class Annotator(ABC):
    """Annotator is an abstract class for annotating a request state. Annotators are used to add additional
    information to a request state that is needed for a metric to understand the request. This could be
    parsing, rendering an image based on the text completion, etc."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def annotate(self, request_state: RequestState) -> List[Dict[str, Any]]:
        """Fills the annotations field of the request state with additional information
        that are implementation specific."""
        pass


@dataclass(frozen=True)
class AnnotatorSpec(ObjectSpec):
    """Specifies how to create an `Annotator`."""

    name: str = "annotator"  # Must be initialized since ObjectSpec has default value for args
    """Name of the annotator. This is used to dispatch to the proper `Annotator`
    in the factory that turn an `AnnotatorSpec` into an `Annotator`."""


class DummyAnnotator(Annotator):
    """A dummy annotator that does nothing."""

    def __init__(self, name: str):
        super().__init__(name)

    def annotate(self, request_state: RequestState) -> List[Dict[str, Any]]:
        if request_state.result is None:
            raise ValueError("Annotation requires a result")
        annotations: List[Dict[str, Any]] = []
        for completion in request_state.result.completions:
            annotations.append({"all_caps": completion.text.upper()})
        return annotations
