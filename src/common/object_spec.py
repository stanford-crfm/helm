from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ObjectSpec:
    """Specifies how to construct an object."""

    # Class name of an object
    class_name: str

    # Arguments used to construct the scenario
    args: Dict[str, Any]


def create_object(spec: ObjectSpec):
    """Create the actual object given the `spec`."""
    # Adapted from https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
    components = spec.class_name.split(".")
    module = __import__(components[0])
    for component in components[1:]:
        module = getattr(module, component)
    return module(**spec.args)
