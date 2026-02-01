"""Tests for schema validation functionality."""

import pytest
from helm.benchmark.presentation.schema import (
    Schema,
    RunGroup,
    MetricGroup,
    MetricNameMatcher,
    Field,
    ValidationSeverity,
    SchemaValidationMessage,
    SchemaValidationError,
    is_template_variable,
    validate_schema,
    validate_schema_file,
    get_all_schema_paths,
)


class TestIsTemplateVariable:
    """Tests for is_template_variable helper function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("${main_name}", True),
            ("${main_split}", True),
            ("${var}", True),
            ("${VAR}", True),
            ("${var_name}", True),
            ("${var_123}", True),
            ("${_private}", True),
            ("", False),
            ("main_name", False),
            ("$main_name", False),
            ("{main_name}", False),
            ("${}", False),
            ("${123}", False),
            ("${main-name}", False),
            ("${main name}", False),
            ("prefix_${var}", False),
            ("${var}_suffix", False),
            ("${var1}${var2}", False),
        ],
    )
    def test_is_template_variable(self, value, expected):
        assert is_template_variable(value) == expected


class TestValidateSchemaValid:
    """Tests for validate_schema on valid schemas."""

    def test_empty_schema(self):
        """An empty schema should validate without errors."""
        schema = Schema()
        messages = validate_schema(schema, strict=True)
        assert len(messages) == 0

    def test_valid_minimal_schema(self):
        """A minimal valid schema should validate without errors."""
        schema = Schema(
            metrics=[Field(name="accuracy", display_name="Accuracy")],
            metric_groups=[
                MetricGroup(
                    name="main_metrics",
                    display_name="Main Metrics",
                    metrics=[MetricNameMatcher(name="accuracy", split="test")],
                )
            ],
            run_groups=[
                RunGroup(
                    name="test_group",
                    display_name="Test Group",
                    metric_groups=["main_metrics"],
                )
            ],
        )
        messages = validate_schema(schema, strict=True)
        assert len(messages) == 0

    def test_valid_schema_with_subgroups(self):
        """A valid schema with subgroups should validate without errors."""
        schema = Schema(
            metrics=[Field(name="accuracy", display_name="Accuracy")],
            metric_groups=[
                MetricGroup(
                    name="main_metrics",
                    display_name="Main Metrics",
                    metrics=[MetricNameMatcher(name="accuracy", split="test")],
                )
            ],
            run_groups=[
                RunGroup(
                    name="parent_group",
                    display_name="Parent Group",
                    subgroups=["child_group"],
                ),
                RunGroup(
                    name="child_group",
                    display_name="Child Group",
                    metric_groups=["main_metrics"],
                ),
            ],
        )
        messages = validate_schema(schema, strict=True)
        assert len(messages) == 0

    def test_valid_schema_with_template_variables(self):
        """Template variables in metrics and splits should be allowed."""
        schema = Schema(
            metric_groups=[
                MetricGroup(
                    name="main_metrics",
                    display_name="Main Metrics",
                    metrics=[MetricNameMatcher(name="${main_name}", split="${main_split}")],
                )
            ],
        )
        messages = validate_schema(schema, strict=True)
        assert len(messages) == 0

    def test_valid_splits(self):
        """All valid split values should be accepted."""
        schema = Schema(
            metrics=[
                Field(name="m1", display_name="M1"),
                Field(name="m2", display_name="M2"),
                Field(name="m3", display_name="M3"),
            ],
            metric_groups=[
                MetricGroup(
                    name="test_group",
                    display_name="Test Group",
                    metrics=[
                        MetricNameMatcher(name="m1", split="test"),
                        MetricNameMatcher(name="m2", split="valid"),
                        MetricNameMatcher(name="m3", split="__all__"),
                    ],
                )
            ],
        )
        messages = validate_schema(schema, strict=True)
        assert len(messages) == 0


class TestValidateSchemaInvalid:
    """Tests for validate_schema on invalid schemas."""

    def test_empty_run_group_name(self):
        """Empty run_group name should produce an error."""
        schema = Schema(
            run_groups=[RunGroup(name="", display_name="Empty Name Group")],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert len(exc_info.value.messages) >= 1
        assert any("empty" in str(m).lower() for m in exc_info.value.messages)

    def test_whitespace_only_name(self):
        """Whitespace-only name should produce an error."""
        schema = Schema(
            run_groups=[RunGroup(name="   ", display_name="Whitespace Name")],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert len(exc_info.value.messages) >= 1

    def test_duplicate_run_group_names(self):
        """Duplicate run_group names should produce an error."""
        schema = Schema(
            run_groups=[
                RunGroup(name="duplicate", display_name="First"),
                RunGroup(name="duplicate", display_name="Second"),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("duplicate" in str(m).lower() for m in exc_info.value.messages)

    def test_undefined_subgroup_reference(self):
        """Referencing an undefined subgroup should produce an error."""
        schema = Schema(
            run_groups=[
                RunGroup(
                    name="parent",
                    display_name="Parent",
                    subgroups=["undefined_child"],
                ),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("undefined_child" in str(m) for m in exc_info.value.messages)

    def test_undefined_metric_group_reference(self):
        """Referencing an undefined metric_group should produce an error."""
        schema = Schema(
            run_groups=[
                RunGroup(
                    name="test_group",
                    display_name="Test",
                    metric_groups=["undefined_metric_group"],
                ),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("undefined_metric_group" in str(m) for m in exc_info.value.messages)

    def test_undefined_metric_in_metric_group(self):
        """Referencing an undefined metric should produce an error."""
        schema = Schema(
            metric_groups=[
                MetricGroup(
                    name="test_metrics",
                    display_name="Test Metrics",
                    metrics=[MetricNameMatcher(name="undefined_metric", split="test")],
                )
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("undefined_metric" in str(m) for m in exc_info.value.messages)

    def test_invalid_split_value(self):
        """Invalid split values should produce an error."""
        schema = Schema(
            metrics=[Field(name="accuracy", display_name="Accuracy")],
            metric_groups=[
                MetricGroup(
                    name="test_metrics",
                    display_name="Test Metrics",
                    metrics=[MetricNameMatcher(name="accuracy", split="invalid_split")],
                )
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("invalid_split" in str(m) for m in exc_info.value.messages)

    def test_circular_subgroup_reference_self(self):
        """Self-referencing subgroup should produce an error."""
        schema = Schema(
            run_groups=[
                RunGroup(name="self_ref", display_name="Self Reference", subgroups=["self_ref"]),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("circular" in str(m).lower() for m in exc_info.value.messages)

    def test_circular_subgroup_reference_chain(self):
        """Chain cycle (A->B->C->A) should produce ONE error, not three."""
        schema = Schema(
            run_groups=[
                RunGroup(name="group_a", display_name="A", subgroups=["group_b"]),
                RunGroup(name="group_b", display_name="B", subgroups=["group_c"]),
                RunGroup(name="group_c", display_name="C", subgroups=["group_a"]),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)

        cycle_errors = [m for m in exc_info.value.messages if "Circular reference" in str(m)]
        assert len(cycle_errors) == 1
        error_str = str(cycle_errors[0])
        assert "group_a" in error_str and "group_b" in error_str and "group_c" in error_str

    def test_undefined_hidden_metric_group(self):
        """Undefined subgroup_metric_groups_hidden should produce an error."""
        schema = Schema(
            run_groups=[
                RunGroup(
                    name="test_group",
                    display_name="Test",
                    subgroup_metric_groups_hidden=["undefined_hidden"],
                ),
            ],
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(schema, strict=True)
        assert any("undefined_hidden" in str(m) for m in exc_info.value.messages)


class TestValidateSchemaWarnings:
    """Tests for validate_schema warning conditions."""

    def test_parent_child_partition_warning(self):
        """Run group with both subgroups and metric_groups should warn."""
        schema = Schema(
            metric_groups=[
                MetricGroup(name="metrics", display_name="Metrics", metrics=[]),
            ],
            run_groups=[
                RunGroup(
                    name="parent",
                    display_name="Parent",
                    subgroups=["child"],
                    metric_groups=["metrics"],
                ),
                RunGroup(name="child", display_name="Child", metric_groups=["metrics"]),
            ],
        )
        messages = validate_schema(schema, strict=False, check_parent_child_partition=True)
        warnings = [m for m in messages if m.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1
        assert any("subgroups and metric_groups" in str(w) for w in warnings)

    def test_orphan_children_warning(self):
        """Orphan child run_groups should warn when check is enabled."""
        schema = Schema(
            metric_groups=[
                MetricGroup(name="metrics", display_name="Metrics", metrics=[]),
            ],
            run_groups=[
                RunGroup(name="orphan_child", display_name="Orphan", metric_groups=["metrics"]),
            ],
        )
        messages = validate_schema(schema, strict=False, check_orphan_children=True)
        warnings = [m for m in messages if m.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1
        assert any("orphan_child" in str(w) for w in warnings)


class TestValidateSchemaNotStrict:
    """Tests for validate_schema with strict=False."""

    def test_non_strict_returns_messages(self):
        """Non-strict mode should return messages but not raise."""
        schema = Schema(
            run_groups=[
                RunGroup(name="parent", display_name="Parent", subgroups=["undefined"]),
            ],
        )
        messages = validate_schema(schema, strict=False)
        assert len(messages) >= 1
        errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1


class TestValidateSchemaFile:
    """Tests for validate_schema_file function."""

    def test_nonexistent_file_strict(self):
        """Nonexistent file should raise in strict mode."""
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_file("/nonexistent/path/schema.yaml", strict=True)
        assert any("not found" in str(m) for m in exc_info.value.messages)

    def test_nonexistent_file_non_strict(self):
        """Nonexistent file should return error message in non-strict mode."""
        messages = validate_schema_file("/nonexistent/path/schema.yaml", strict=False)
        assert len(messages) >= 1
        assert any("not found" in str(m) for m in messages)


class TestPackagedSchemaFiles:
    """Tests to validate all packaged schema files."""

    def test_all_packaged_schemas_can_be_validated(self):
        """All packaged schema files should be loadable and validatable.

        This test ensures:
        1. All schema files can be loaded without YAML/parsing errors
        2. Pre-existing validation issues are tracked but don't fail the test

        If this test fails, it means a schema file has a fundamental loading
        problem (not just validation warnings about references).
        """
        schema_paths = get_all_schema_paths()
        assert len(schema_paths) > 0, "No schema files found"

        load_failures = []
        total_validation_errors = 0
        total_warnings = 0

        for schema_path in schema_paths:
            messages = validate_schema_file(schema_path, strict=False)
            errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
            warnings = [m for m in messages if m.severity == ValidationSeverity.WARNING]

            # Check for critical loading failures (not validation errors)
            for msg in messages:
                msg_str = str(msg)
                if any(
                    phrase in msg_str
                    for phrase in [
                        "Invalid YAML syntax",
                        "Failed to load schema",
                        "Schema file not found",
                    ]
                ):
                    load_failures.append(msg)

            total_validation_errors += len(errors)
            total_warnings += len(warnings)

        # Fail on loading/parsing errors - these are always regressions
        assert len(load_failures) == 0, "Schema files failed to load:\n" + "\n".join(
            f"  - {msg}" for msg in load_failures
        )

        # Log summary for informational purposes
        print(
            f"\nSchema validation summary: {len(schema_paths)} files, "
            f"{total_validation_errors} validation errors, {total_warnings} warnings"
        )


class TestSchemaValidationMessage:
    """Tests for SchemaValidationMessage."""

    def test_message_str_full(self):
        """Full message should include all parts."""
        msg = SchemaValidationMessage(
            severity=ValidationSeverity.ERROR,
            message="Test error message",
            schema_path="/path/to/schema.yaml",
            location="run_groups[test]",
        )
        str_msg = str(msg)
        assert "[/path/to/schema.yaml]" in str_msg
        assert "[ERROR]" in str_msg
        assert "at run_groups[test]:" in str_msg
        assert "Test error message" in str_msg

    def test_message_str_minimal(self):
        """Minimal message should work without optional fields."""
        msg = SchemaValidationMessage(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
        )
        str_msg = str(msg)
        assert "[WARNING]" in str_msg
        assert "Test warning" in str_msg
