from helm.benchmark.metrics.ts_guessing_contamination_metrics import TSGuessingMetric

from helm.benchmark.scenarios.ts_guessing_contamination.prompt_translations import (
    TS_GUESSING_BASE,
    TS_GUESSING_MULTICHOICE,
)

PROMPT_CONFIGS = {
    "ts_guessing_question_base": TS_GUESSING_BASE,
    "ts_guessing_question_multichoice": TS_GUESSING_MULTICHOICE,
}


def test_ts_guessing_metric_clean_processing_english():
    language = "en"
    strategy = "ts_guessing_question_base"
    metric = TSGuessingMetric(language, strategy)

    # Test removing English prefixes
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} Rio") == "rio"

    # Test removing quotes
    assert metric._process_response('"Rio"') == "rio"
    assert metric._process_response("'Brasilia'") == "brasilia"

    # Test extracting only the first sentence
    assert metric._process_response("Paris. The rest of the text is ignored.") == "paris."

    # Test removing [MASK] tags
    assert metric._process_response("[MASK] Tokyo") == "tokyo"


def test_ts_guessing_metric_clean_processing_portuguese():
    language = "pt"
    strategy = "ts_guessing_question_base"
    metric = TSGuessingMetric(language, strategy)

    # Test removing Portuguese prefixes
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} Recife") == "recife"

    # Ensure normal stripping and lowering works
    assert metric._process_response("  Curitiba  ") == "curitiba"
