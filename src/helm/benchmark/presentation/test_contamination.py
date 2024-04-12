from helm.benchmark.presentation.schema import read_schema, get_default_schema_path
from helm.benchmark.presentation.contamination import read_contamination, validate_contamination


def test_contamination_schema():
    schema = read_schema(get_default_schema_path())
    contamination = read_contamination()
    validate_contamination(contamination, schema)

    assert contamination.get_point("openai/davinci", "boolq") is not None
    assert contamination.get_point("openai/davinci", "not_exists") is None
