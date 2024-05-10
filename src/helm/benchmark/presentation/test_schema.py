from helm.benchmark.presentation.schema import get_adapter_fields


def test_get_adapter_fields() -> None:
    adapter_fields = get_adapter_fields()
    assert adapter_fields
    assert adapter_fields[0].name == "method"
    assert (
        adapter_fields[0].description
        == "The high-level strategy for converting instances into a prompt for the language model."
    )
