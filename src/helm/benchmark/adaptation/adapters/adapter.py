from abc import ABC, abstractmethod
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory


class Adapter(ABC):
    """
    An `Adapter`, guided by the `AdapterSpec`, takes a `Scenario` and produces
    a `ScenarioState`. This is where prompt hacking logic should live.
    """

    def __init__(self, adapter_spec: AdapterSpec, tokenizer_service: TokenizerService):
        self.adapter_spec: AdapterSpec = adapter_spec
        self.window_service: WindowService = WindowServiceFactory.get_window_service(
            adapter_spec.model_deployment, tokenizer_service
        )

    @abstractmethod
    def adapt(self, instances: List[Instance], parallelism: int) -> List[RequestState]:
        """
        Takes a a list of `Instance`s and returns a `ScenarioState` with the
        list of corresponding `RequestState`s.
        """
        pass
