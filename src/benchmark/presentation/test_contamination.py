from benchmark.presentation.schema import read_schema
from contamination import read_contamination, validate_contamination


def test_contamination_schema():
    schema = read_schema()
    contamination = read_contamination()
    validate_contamination(contamination, schema)

    assert contamination.get_point("not_exists", "openai/davinci") is None
    assert contamination.get_point("boolq", "openai/davinci") is not None
