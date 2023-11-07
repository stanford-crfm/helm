import importlib
import dataclasses
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Hashable, Type, TypeVar


@dataclass(frozen=True)
class ObjectSpec:
    """Specifies how to construct an object."""

    # Class name of an object
    class_name: str

    # Arguments used to construct the scenario
    args: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        def get_arg_value(key: str) -> Any:
            value = self.args[key]
            # Convert non hashable objects into string
            if not isinstance(value, Hashable):
                return value.__str__()
            return value

        args_tuple = tuple((k, get_arg_value(k)) for k in sorted(self.args.keys()))
        return hash((self.class_name, args_tuple))


def get_class_by_name(full_class_name: str) -> Type[Any]:
    components = full_class_name.split(".")
    class_name = components[-1]
    module_name = ".".join(components[:-1])
    return getattr(importlib.import_module(module_name), class_name)


ObjectSpecT = TypeVar("ObjectSpecT", bound=ObjectSpec)


def inject_object_spec_args(
    spec: ObjectSpecT,
    constant_bindings: Optional[Dict[str, Any]] = None,
    provider_bindings: Optional[Dict[str, Callable[[], Any]]] = None,
) -> ObjectSpecT:
    """Return a new ObjectSpec that is a copy of the original ObjectSpec with additional arguments.

    The original ObjectSpec may be missing arguments for parameters that are required by the
    ObjectSpec's class's constructor.
    This function returns a new ObjectSpec with these missing parameter filled in.
    To do this, for every missing parameter, check look up each of the `*_bindings` arguments in order until we
    find one with a key matching the missing parameter's name.
    If found in constant_bindings, add the corresponding value to args.
    If found in provider_bindings, call the corresponding value and add the return values to args.

    This is loosely based on instance (constant) bindings and provider bindings in Guice dependency injection.

    Example:

    class MyClass:
        def __init__(a: int, b: int, c: int, d: int = 0):
            pass

    old_object_spec = ObjectSpec(class_name="MyClass", args={"a": 11})
    new_object_spec = inject_object_spec_args(old_object_spec, {"b": 12}, {"c": lambda: 13})
    # new_object_spec is now ObjectSpec(class_name="MyClass", args={"a": 11, "b": 12, "c": 13})
    """
    cls = get_class_by_name(spec.class_name)
    init_signature = inspect.signature(cls.__init__)
    args = {}
    args.update(spec.args)
    for parameter_name in init_signature.parameters.keys():
        if parameter_name == "self" or parameter_name in args:
            continue
        elif constant_bindings and parameter_name in constant_bindings:
            args[parameter_name] = constant_bindings[parameter_name]
        elif provider_bindings and parameter_name in provider_bindings:
            args[parameter_name] = provider_bindings[parameter_name]()
    return dataclasses.replace(spec, args=args)


def create_object(spec: ObjectSpec):
    """Create the actual object given the `spec`."""
    cls = get_class_by_name(spec.class_name)
    args = {}
    args.update(spec.args)
    return cls(**args)


def parse_object_spec(description: str) -> ObjectSpec:
    """
    Parse `description` into an `ObjectSpec`.
    `description` has the format:
        <class_name>:<key>=<value>,<key>=<value>
    Usually, the description is something that's succinct and can be typed on the command-line.
    Here, value defaults to string.
    """

    def parse_arg(arg: str) -> Tuple[str, Any]:
        if "=" not in arg:
            raise ValueError(f"Expected <key>=<value>, got '{arg}'")
        value: Any
        key, value = arg.split("=", 1)

        # Try to convert to number
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

        return (key, value)

    if ":" in description:
        name, args_str = description.split(":", 1)
        args: Dict[str, Any] = dict(parse_arg(arg) for arg in args_str.split(","))
    else:
        name = description
        args = {}
    return ObjectSpec(name, args)
