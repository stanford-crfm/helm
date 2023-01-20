from abc import ABC, abstractmethod
from typing import List

import numpy as np

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Instance, TRAIN_SPLIT, EVAL_SPLITS
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.hierarchical_logger import hlog


class Adapter(ABC):
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`. This is where prompt hacking logic should live.
    """

    def __init__(self, adapter_spec: AdapterSpec, tokenizer_service: TokenizerService):
        self.adapter_spec: AdapterSpec = adapter_spec
        self.window_service: WindowService = WindowServiceFactory.get_window_service(
            adapter_spec.model, tokenizer_service
        )

    @abstractmethod
    def adapt(self, instances: List[Instance], parallelism: int) -> ScenarioState:
        """
        Takes a a list of `Instance`s and returns a `ScenarioState` with the
        list of corresponding `RequestState`s.
        """
        pass

    @abstractmethod
    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        """
        Given a validation or test `Instance`, generates one or more `RequestState`s.
        """
        pass

    def get_run_instances(self, instances: List[Instance]) -> List[Instance]:
        """
        Get the instances necessary for this run:
        Train instances (split=train): keep all (if any) for in-context learning
        Eval instances (split=valid or test): keep at most `max_eval_instances` specified in `AdapterSpec` by sampling
        Return the resulting train and eval instances.
        """
        all_train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]

        all_eval_instances: List[Instance] = [instance for instance in instances if instance.split in EVAL_SPLITS]
        if (
            self.adapter_spec.max_eval_instances is not None
            and len(all_eval_instances) > self.adapter_spec.max_eval_instances
        ):
            # Pick the first `self.adapter_spec.max_eval_instances`.
            # The random sampling includes instances monotonically.
            np.random.seed(0)
            selected_eval_instances = list(
                np.random.choice(
                    all_eval_instances,  # type: ignore
                    self.adapter_spec.max_eval_instances,
                    replace=False,
                )
            )
        else:
            selected_eval_instances = all_eval_instances

        hlog(
            f"{len(instances)} instances, "
            f"{len(all_train_instances)} train instances, "
            f"{len(selected_eval_instances)}/{len(all_eval_instances)} eval instances"
        )

        return all_train_instances + selected_eval_instances
