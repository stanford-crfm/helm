from typing import Dict, Any
from abc import abstractmethod, ABC

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.object_spec import ObjectSpec


class Annotator(ABC):
    """Annotator is an abstract class for annotating a request state. Annotators are used to add additional
    information to a request state that is needed for a metric to understand the request. This could be
    parsing, rendering an image based on the text completion, etc."""

    @abstractmethod
    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        """Fills the annotations field of the request state with additional information
        that are implementation specific."""
        pass


class AnnotatorSpec(ObjectSpec):
    """Specifies how to create an `Annotator`."""

    name: str
    """Name of the annotator. This is used to dispatch to the proper `Annotator`
    in the factory that turn an `AnnotatorSpec` into an `Annotator`."""
