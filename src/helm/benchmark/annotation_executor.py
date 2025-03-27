import os

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, replace
from helm.common.cache_backend_config import (
    CacheBackendConfig,
    BlackHoleCacheBackendConfig,
    MongoCacheBackendConfig,
    SqliteCacheBackendConfig,
)

from helm.common.general import ensure_directory_exists, parallel_map, get_credentials
from helm.common.hierarchical_logger import htrack, hlog
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.annotation.annotator_factory import AnnotatorFactory
from helm.proxy.services.service import CACHE_DIR


class AnnotationExecutorError(Exception):
    pass


@dataclass(frozen=True)
class AnnotationExecutionSpec:

    local_path: str
    """Path where API credentials and cache is stored.

    This path is the same as `--base-path` when launching the proxy server (see server.py).
    Required when url is not set."""

    parallelism: int
    """How many threads to have at once"""

    dry_run: bool = False
    """Whether to skip execution"""

    sqlite_cache_backend_config: Optional[SqliteCacheBackendConfig] = None
    """If set, SQLite will be used for the cache.

    This specifies the directory in which the SQLite cache will store files.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""

    mongo_cache_backend_config: Optional[MongoCacheBackendConfig] = None
    """If set, MongoDB will be used for the cache.

    This specifies the MongoDB database to be used by the MongoDB cache.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""


class AnnotationExecutor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: AnnotationExecutionSpec):
        self.execution_spec = execution_spec

        cache_backend_config: CacheBackendConfig
        if execution_spec.sqlite_cache_backend_config and execution_spec.mongo_cache_backend_config:
            raise AnnotationExecutorError(
                "At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."
            )
        elif execution_spec.sqlite_cache_backend_config:
            cache_backend_config = execution_spec.sqlite_cache_backend_config
        elif execution_spec.mongo_cache_backend_config:
            cache_backend_config = execution_spec.mongo_cache_backend_config
        else:
            cache_backend_config = BlackHoleCacheBackendConfig()

        base_path: str = execution_spec.local_path
        ensure_directory_exists(base_path)
        client_file_storage_path = os.path.join(base_path, CACHE_DIR)
        ensure_directory_exists(client_file_storage_path)
        credentials: Dict[str, str] = get_credentials(base_path)
        self.factory = AnnotatorFactory(
            credentials=credentials,
            file_storage_path=client_file_storage_path,
            cache_backend_config=cache_backend_config,
        )

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped annotation.")
            return scenario_state

        if not scenario_state.annotator_specs:
            hlog("No annotators to run.")
            return scenario_state

        try:
            annotators: List[Annotator] = [
                self.factory.get_annotator(annotator_spec) for annotator_spec in scenario_state.annotator_specs
            ]
        except Exception as e:
            raise AnnotationExecutorError(f"Could not initialize annotator for spec: {str(e)} ") from e

        if all(getattr(annotator, "use_global_metric", False) for annotator in annotators):
            # Do it!
            request_states = self.process_all(
                annotators, scenario_state.request_states  # processing all request together
            )

        else:
            # Do it!
            def do_it(request_state: RequestState) -> RequestState:
                return self.process(annotators, request_state)

            request_states = parallel_map(
                do_it,
                scenario_state.request_states,
                parallelism=self.execution_spec.parallelism,
            )

        hlog(f"Annotated {len(request_states)} requests")
        return ScenarioState(
            adapter_spec=scenario_state.adapter_spec,
            request_states=request_states,
            annotator_specs=scenario_state.annotator_specs,
        )

    def process(self, annotators: List[Annotator], state: RequestState) -> RequestState:
        annotations: Dict[str, Any] = {}
        try:
            for annotator in annotators:
                new_annotations = annotator.annotate(state)
                annotations[annotator.name] = new_annotations
        except Exception as e:
            raise AnnotationExecutorError(f"{str(e)} Request: {state.request}") from e
        return replace(state, annotations=annotations)

    def process_all(self, annotators: List[Annotator], states: List[RequestState]) -> List[RequestState]:
        annotations: Dict[str, Any] = {}
        try:
            for annotator in annotators:
                new_annotations = annotator.annotate_all(states)
                annotations[annotator.name] = new_annotations
        except Exception as e:
            raise AnnotationExecutorError(f"{str(e)} Request: {[state.request for state in states]}") from e
        return [replace(state, annotations=new_annotations[idx]) for idx, state in enumerate(states)]
