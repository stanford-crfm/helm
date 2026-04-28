from helm.common.hierarchical_logger import hlog

from helm.benchmark.scenarios.contamination.ts_guessing_question_based import TSGuessingQuestionBaseStrategy
from helm.benchmark.scenarios.contamination.ts_guessing_question_multichoice import (
    TSGuessingQuestionMultiChoiceStrategy,
)

REGISTERED_STRATEGIES = {
    TSGuessingQuestionBaseStrategy.STRATEGY_NAME: TSGuessingQuestionBaseStrategy,
    TSGuessingQuestionMultiChoiceStrategy.STRATEGY_NAME: TSGuessingQuestionMultiChoiceStrategy,
}


def get_contamination_strategy(strategy_name: str):
    """
    Returns an instance of the requested contamination strategy.
    """
    strategy_class = REGISTERED_STRATEGIES.get(strategy_name)
    if strategy_class:
        return strategy_class()
    else:
        hlog(
            f"ERROR: Unknown contamination strategy: '{strategy_name}'. "
            f"Available: {list(REGISTERED_STRATEGIES.keys())}"
        )
        return None
