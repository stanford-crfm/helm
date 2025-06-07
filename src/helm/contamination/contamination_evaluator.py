from abc import ABC, abstractmethod
from typing import List

from helm.benchmark.executor import Executor
from helm.benchmark.metrics.metric import Stat
from helm.benchmark.runner import ScenarioState
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class ContaminationEvaluator(ABC):
    """
    Abstract base class for contamination evaluation strategies.
    Concrete implementations must provide the logic for a specific evaluation method.

    Subclasses are expected to define a class-level attribute 'STRATEGY_NAME'.
    """

    STRATEGY_NAME: str

    @abstractmethod
    def evaluate(
        self,
        executor: Executor,
        benchmark_path: str,
        scenario_state: ScenarioState,
        language: str,
        tokenizer_service: TokenizerService,
    ) -> List[Stat]:
        """
        Evaluates contamination according to the specific strategy implementation.
        """
        pass
