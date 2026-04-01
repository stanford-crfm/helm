import pytest

from helm.benchmark.metrics.contamination_metrics import TSGuessingMetric

def test_ts_guessing_metric_clean_processing_english():
    metric = TSGuessingMetric(language="en")

    # Test removing English prefixes
    assert metric._process_response("Answer: Rio") == "rio"
    assert metric._process_response("The answer is: Brasilia") == "brasilia"

    # Test removing quotes
    assert metric._process_response('"Rio"') == "rio"
    assert metric._process_response("'Brasilia'") == "brasilia"

    # Test extracting only the first sentence
    assert metric._process_response("Paris. The rest of the text is ignored.") == "paris."

    # Test removing [MASK] tags
    assert metric._process_response("[MASK] Tokyo") == "tokyo"

def test_ts_guessing_metric_clean_processing_portuguese():
    metric = TSGuessingMetric(language="pt")

    # Test removing Portuguese prefixes (A GRANDE SACADA)
    assert metric._process_response("Resposta: Recife") == "recife"
    assert metric._process_response("A resposta é: São Paulo") == "são paulo"
    assert metric._process_response("resposta: salvador") == "salvador"

    # Ensure normal stripping and lowering works
    assert metric._process_response("  Curitiba  ") == "curitiba"