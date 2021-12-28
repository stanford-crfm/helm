from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ObjectSpec:
    class_name: str  # Class name of an object
    args: Dict[str, Any]  # Arguments used to construct the scenario


def create_object(spec: ObjectSpec):
    # Adapted from https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
    components = spec.class_name.split(".")
    module = __import__(components[0])
    for component in components[1:]:
        module = getattr(module, component)
    return module(**spec.args)
