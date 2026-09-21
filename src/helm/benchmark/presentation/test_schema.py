from importlib import resources

from helm.benchmark.presentation.schema import (
    SCHEMA_YAML_PACKAGE,
    get_adapter_fields,
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


def test_validate_built_in_schemas() -> None:
    """Test that all schemas in the repository pass validation"""
    for traversable in resources.files(SCHEMA_YAML_PACKAGE).iterdir():
        if traversable.is_file() and traversable.name.startswith("schema_") and traversable.name.endswith(".yaml"):
            schema_path = str(traversable)
            schema = read_schema(schema_path)
            messages = validate_schema(schema, schema_path=schema_path, strict=False)
            errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]

            assert len(errors) == 0, f"Schema validation errors: {errors}"
