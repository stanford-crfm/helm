from typing import Optional
from helm.common.hierarchical_logger import hlog

from helm.contamination.contamination_evaluator import ContaminationEvaluator
from helm.contamination.ts_guessing_question_based import TSGuessingQuestionBasedContaminationEvaluator
from helm.contamination.ts_guessing_question_multichoice import TSGuessingQuestionMultiChoiceContaminationEvaluator

REGISTERED_EVALUATORS = dict(
    [
        (
            TSGuessingQuestionBasedContaminationEvaluator.STRATEGY_NAME,
            TSGuessingQuestionBasedContaminationEvaluator,
        ),
        (
            TSGuessingQuestionMultiChoiceContaminationEvaluator.STRATEGY_NAME,
            TSGuessingQuestionMultiChoiceContaminationEvaluator,
        ),
    ]
)


def get_contamination_evaluator(method_name: str) -> Optional[ContaminationEvaluator]:
    """
    Returns an instance of the requested contamination evaluation strategy, if available.
    """
    evaluator_class = REGISTERED_EVALUATORS.get(method_name)
    if evaluator_class:
        return evaluator_class()
    else:
        hlog(
            f"CONTAMINATION_EVALUATOR_FACTORY ERROR: Unknown contamination evaluation method: '{method_name}'. "
            f"Available methods: {list(REGISTERED_EVALUATORS.keys())}"
        )
        return None
