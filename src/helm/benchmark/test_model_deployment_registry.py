"""Tests for register_model_deployment merge logic (issue #3709)."""
import pytest

from helm.benchmark.model_deployment_registry import (
    ALL_MODEL_DEPLOYMENTS,
    DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT,
    ClientSpec,
    ModelDeployment,
    register_model_deployment,
)


def _make_deployment(
    name: str = "test/model",
    tokenizer_name: str | None = None,
    max_sequence_length: int | None = None,
    deprecated: bool = False,
) -> ModelDeployment:
    return ModelDeployment(
        name=name,
        client_spec=ClientSpec(class_name="helm.clients.test.TestClient"),
        tokenizer_name=tokenizer_name,
        max_sequence_length=max_sequence_length,
        deprecated=deprecated,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test: remove any deployment named 'test/model' before and after."""
    _purge("test/model")
    yield
    _purge("test/model")


def _purge(name: str) -> None:
    DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT.pop(name, None)
    to_remove = [d for d in ALL_MODEL_DEPLOYMENTS if d.name == name]
    for d in to_remove:
        ALL_MODEL_DEPLOYMENTS.remove(d)


class TestRegisterModelDeploymentMerge:
    def test_first_registration_stored_as_is(self):
        d = _make_deployment(tokenizer_name="tok/v1", max_sequence_length=4096)
        register_model_deployment(d)
        stored = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"]
        assert stored.tokenizer_name == "tok/v1"
        assert stored.max_sequence_length == 4096

    def test_override_with_full_entry_replaces_all_fields(self):
        # Built-in config registered first.
        register_model_deployment(_make_deployment(tokenizer_name="tok/v1", max_sequence_length=4096))
        # prod_env override with complete config registered second.
        register_model_deployment(_make_deployment(tokenizer_name="tok/v2", max_sequence_length=8192))
        stored = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"]
        assert stored.tokenizer_name == "tok/v2"
        assert stored.max_sequence_length == 8192

    def test_sparse_override_inherits_none_fields_from_builtin(self):
        # Regression for issue #3709: prod_env entry has tokenizer_name=None
        # (i.e., the field was never set in the prod_env YAML).  Before the fix
        # this would shadow the built-in tokenizer_name with None.
        register_model_deployment(_make_deployment(tokenizer_name="tok/v1", max_sequence_length=4096))
        # prod_env entry omits tokenizer_name.
        register_model_deployment(_make_deployment(tokenizer_name=None, max_sequence_length=8192))
        stored = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"]
        # tokenizer_name must be inherited from the built-in entry.
        assert stored.tokenizer_name == "tok/v1"
        # max_sequence_length must be taken from the prod_env entry.
        assert stored.max_sequence_length == 8192

    def test_sparse_override_none_max_sequence_inherits_from_builtin(self):
        register_model_deployment(_make_deployment(tokenizer_name="tok/v1", max_sequence_length=4096))
        # prod_env entry sets tokenizer but not max_sequence_length.
        register_model_deployment(_make_deployment(tokenizer_name="tok/v2", max_sequence_length=None))
        stored = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"]
        assert stored.tokenizer_name == "tok/v2"
        assert stored.max_sequence_length == 4096

    def test_deprecated_always_taken_from_new_entry(self):
        register_model_deployment(_make_deployment(deprecated=False))
        register_model_deployment(_make_deployment(deprecated=True))
        stored = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"]
        assert stored.deprecated is True

    def test_only_one_entry_in_lookup_dict_after_override(self):
        register_model_deployment(_make_deployment(tokenizer_name="tok/v1"))
        register_model_deployment(_make_deployment(tokenizer_name="tok/v2"))
        # The dict must hold exactly one entry for this name.
        assert DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT["test/model"].tokenizer_name == "tok/v2"
