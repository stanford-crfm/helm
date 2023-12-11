from helm.benchmark.presentation.schema import read_schema, SCHEMA_CLASSIC_YAML_FILENAME
from helm.benchmark.presentation.contamination import read_contamination, validate_contamination


def test_contamination_schema():
    schema = read_schema(SCHEMA_CLASSIC_YAML_FILENAME)
    contamination = read_contamination()
    validate_contamination(contamination, schema)

    assert contamination.get_point("openai/davinci", "boolq") is not None
    assert contamination.get_point("openai/davinci", "not_exists") is None
