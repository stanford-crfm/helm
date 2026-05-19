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

    # Test removing English prefixes in base strategy
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} Rio") == "rio"
    assert metric._process_response(f"{prefix}Brasília") == "brasília"

    # Test removind prefix in multichoice strategy
    strategy = "ts_guessing_question_multichoice"
    metric = TSGuessingMetric(language, strategy)
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} Rio") == "rio"
    assert metric._process_response(f"{prefix}Brasília") == "brasília"


def test_ts_guessing_metric_clean_processing_portuguese():
    language = "pt"
    strategy = "ts_guessing_question_base"
    metric = TSGuessingMetric(language, strategy)

    # Test removing Portuguese prefixes in base strategy
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} Recife") == "recife"
    assert metric._process_response(f"{prefix}Berlin") == "berlin"

    # Test removing prefix in multichoice strategy
    strategy = "ts_guessing_question_multichoice"
    metric = TSGuessingMetric(language, strategy)
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix} London") == "london"


def test_ts_guessing_metric_clean_processing_chinese():
    language = "zh"
    strategy = "ts_guessing_question_base"
    metric = TSGuessingMetric(language, strategy)

    # Test removing Chinese prefixes in base strategy
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]
    assert metric._process_response(f"{prefix}北京") == "北京"
    assert metric._process_response(f"{prefix} 上海") == "上海"

    # Test removing prefix in multichoice strategy
    strategy = "ts_guessing_question_multichoice"
    metric = TSGuessingMetric(language, strategy)
    prefix = PROMPT_CONFIGS[strategy][language]["answer_prefix"]

    assert metric._process_response(f"{prefix}广州") == "广州"
    assert metric._process_response(f"{prefix} 深圳") == "深圳"


def test_ts_guessing_metric_clean_processing_general():
    language = "en"
    strategy = "ts_guessing_question_base"
    metric = TSGuessingMetric(language, strategy)

    # Test removing choice letter prefixes in the beginning of the text
    assert metric._process_response("A: Madrid") == "madrid"
    assert metric._process_response("B. Lisbon") == "lisbon"
    assert metric._process_response("C - Rome") == "rome"
    assert metric._process_response("D) Berlin") == "berlin"

    # Ensure normal stripping and lowering works
    assert metric._process_response("  Curitiba  ") == "curitiba"

    # Test removing quotes
    assert metric._process_response('"Rio"') == "rio"
    assert metric._process_response("'Brasilia'") == "brasilia"

    # Test extracting only the first sentence
    assert metric._process_response("Paris. The rest of the text is ignored.") == "paris."
    assert metric._process_response("Berlin is the capital. Additional info.") == "berlin is the capital."

    # Test removing [MASK] tags
    assert metric._process_response("[MASK] Tokyo") == "tokyo"


def test_ts_guessing_metric_invalid_strategy():
    try:
        metric = TSGuessingMetric(language="en", strategy="invalid_strategy")
        metric._process_response("It has to raise an error for unsupported strategy")
        assert False, "Expected ValueError for invalid strategy"
    except ValueError as e:
        assert "Unknown strategy" in str(e)


def test_ts_guessing_metric_unsupported_language():
    try:
        metric = TSGuessingMetric(language="fr", strategy="ts_guessing_question_base")
        metric._process_response("It has to raise an error for unsupported language")
        assert False, "Expected ValueError for unsupported language"
    except ValueError as e:
        assert "Language 'fr' not supported" in str(e)
