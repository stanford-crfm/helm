import traceback

from helm.contamination.ts_guessing_question_based import TSGuessingQuestionBasedContaminationEvaluator
from helm.contamination.ts_guessing_question_multichoice import TSGuessingQuestionMultiChoiceContaminationEvaluator
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.runner import ScenarioState
from helm.benchmark.executor import Executor
from helm.benchmark.metrics.metric import Stat
from helm.common.hierarchical_logger import hlog
from typing import List, Any


class ContaminationEvaluator:
    """
    Class responsible for dispatching to and executing different contamination
    evaluation strategies based on the specified method.
    """

    def evaluate(
        self,
        executor: Executor,
        method: str,
        benchmark_path: str,
        scenario_state: ScenarioState,
        language: str,
        tokenizer_service: TokenizerService,
    ) -> List[Stat]:
        """
        Selects and runs the appropriate contamination evaluation strategy.

        Args:
            executor: The HELM executor for running model queries.
            method: The contamination evaluation method/strategy to use (e.g., "base", "multichoice").
            benchmark_path: Path to the benchmark data or relevant context.
            scenario_state: The current scenario state (should be a deep copy).
            language: Defines the language for prompts and language-specific processing.
            tokenizer_service: The service for tokenizing text.

        Returns:
            A list of dictionaries, where each dictionary represents a PinnedStat
            (serializable version of Stat) for contamination metrics.
            Returns an empty list if the method is unknown or evaluation fails.
        """
        evaluator: Any

        if method == TSGuessingQuestionBasedContaminationEvaluator.STRATEGY_NAME:
            evaluator = TSGuessingQuestionBasedContaminationEvaluator()
        elif method == TSGuessingQuestionMultiChoiceContaminationEvaluator.STRATEGY_NAME:
            evaluator = TSGuessingQuestionMultiChoiceContaminationEvaluator()
        else:
            hlog(
                f"CONTAMINATION_EVALUATOR ERROR: Unknown contamination evaluation method specified: '{method}'. "
                "Available methods depend on registered strategy classes and their STRATEGY_NAME."
            )
            return []

        # Run the selected evaluator
        try:
            return evaluator.evaluate(
                executor=executor,
                benchmark_path=benchmark_path,
                scenario_state=scenario_state,
                language=language,
                tokenizer_service=tokenizer_service,
            )
        except Exception as e:
            hlog(
                f"CONTAMINATION_EVALUATOR CRITICAL: An error occurred while running the '{method}' contamination strategy: {e}\n{traceback.format_exc()}"
            )
            return []
