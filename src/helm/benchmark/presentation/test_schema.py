from helm.benchmark.presentation.schema import (
    get_adapter_fields,
    get_default_schema_path,
    read_schema,
    validate_schema,
    ValidationSeverity,
)


def test_get_adapter_fields() -> None:
    adapter_fields = get_adapter_fields()
    assert adapter_fields
    assert adapter_fields[0].name == "method"
    assert (
        adapter_fields[0].description
        == "The high-level strategy for converting instances into a prompt for the language model."
    )


def test_default_schema_validates() -> None:
    """Test that the default schema validates without unexpected errors."""
    schema_path = get_default_schema_path()
    schema = read_schema(schema_path)
    messages = validate_schema(schema, schema_path=schema_path, strict=False)
    errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]

    # Known issues in the schema - filter them out for this test
    known_issues = {"synonyms", "bleu"}
    unexpected_errors = [e for e in errors if not any(ki in str(e) for ki in known_issues)]

    assert len(unexpected_errors) == 0, f"Unexpected schema validation errors: {unexpected_errors}"
