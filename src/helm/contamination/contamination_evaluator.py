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
        """
        evaluator: Any

        # Select the evaluation class based on the provided method string.
        if method == TSGuessingQuestionBasedContaminationEvaluator.STRATEGY_NAME:
            evaluator = TSGuessingQuestionBasedContaminationEvaluator()
        elif method == TSGuessingQuestionMultiChoiceContaminationEvaluator.STRATEGY_NAME:
            evaluator = TSGuessingQuestionMultiChoiceContaminationEvaluator()
        else:
            # Log an error if the specified method is unknown.
            hlog(
                f"CONTAMINATION_EVALUATOR ERROR: Unknown contamination evaluation method specified: '{method}'. "
                "Available methods depend on registered strategy classes and their STRATEGY_NAME."
            )
            return []

        # Attempt to run the selected evaluator.
        try:
            # Execute the evaluation using the chosen strategy.
            # Parameters (executor, benchmark_path, etc.) are passed through
            # to the specific strategy's 'evaluate' method.
            return evaluator.evaluate(
                executor=executor,
                benchmark_path=benchmark_path,
                scenario_state=scenario_state,
                language=language,
                tokenizer_service=tokenizer_service,
            )
        except Exception as e:
            # Catch and log any critical exceptions during the evaluation.
            # This ensures that an error within one strategy doesn't stop the entire process.
            hlog(
                f"CONTAMINATION_EVALUATOR CRITICAL: An error occurred while running the '{method}'"
                f" contamination strategy: {e}\n{traceback.format_exc()}"
            )
            return []
