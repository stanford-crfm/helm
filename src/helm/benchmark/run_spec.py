from dataclasses import dataclass, field
import importlib
import os
import pkgutil
from typing import Callable, Dict, Iterable, List, Optional, TypeVar

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.augmentations.data_augmenter import DataAugmenterSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of stats based on the defined metrics.
    """

    name: str
    """Unique identifier of the RunSpec"""

    scenario_spec: ScenarioSpec
    """Which scenario"""

    adapter_spec: AdapterSpec
    """Specifies how to adapt an instance into a set of requests"""

    metric_specs: List[MetricSpec]
    """What to evaluate on"""

    data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec()
    """Data augmenter. The default `DataAugmenterSpec` does nothing."""

    groups: List[str] = field(default_factory=list)
    """Groups that this run spec belongs to (for aggregation)"""

    annotators: Optional[List[AnnotatorSpec]] = None
    """Annotators to use for this run spec"""

    def __post_init__(self):
        """
        `self.name` is used as the name of the output folder for the `RunSpec`.
        Clean up `self.name` by replacing any "/"'s with "_".
        """
        # TODO: Don't mutate name! clean this up before passing it into the constructor here
        object.__setattr__(self, "name", self.name.replace(os.path.sep, "_"))


RunSpecFunction = Callable[..., RunSpec]


_REGISTERED_RUN_SPEC_FUNCTIONS: Dict[str, RunSpecFunction] = {}
"""Dict of run spec function names to run spec functions."""


F = TypeVar("F", bound=RunSpecFunction)


def run_spec_function(name: str) -> Callable[[F], F]:
    """Register the run spec function under the given name."""

    def wrap(func: F) -> F:
        if name in _REGISTERED_RUN_SPEC_FUNCTIONS:
            raise ValueError(f"A run spec function with name {name} already exists")
        _REGISTERED_RUN_SPEC_FUNCTIONS[name] = func
        return func

    return wrap


# Copied from https://docs.python.org/3/library/pkgutil.html#pkgutil.iter_modules
def _iter_namespace(ns_pkg) -> Iterable[pkgutil.ModuleInfo]:
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def discover_run_spec_functions() -> None:
    """Discover and register all run spec functions under helm.benchmark.run_specs"""
    import helm.benchmark.run_specs  # noqa

    for finder, name, ispkg in _iter_namespace(helm.benchmark.run_specs):
        importlib.import_module(name)


def get_run_spec_function(name: str) -> Optional[RunSpecFunction]:
    """Return the run spec function registered under the given name."""
    discover_run_spec_functions()
    return _REGISTERED_RUN_SPEC_FUNCTIONS.get(name)
