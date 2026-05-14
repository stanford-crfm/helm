import os
import tempfile

import pytest

from helm.common.cache import MongoCacheConfig
from helm.common.cache_backend_config import (
    MongoCacheBackendConfig,
    get_mongo_cache_backend_config,
    redact_mongo_uri,
)


def test_redact_mongo_uri_removes_password():
    uri = "mongodb://user:password@example.com:27017/cache?retryWrites=true"

    assert redact_mongo_uri(uri) == "mongodb://example.com:27017/cache?retryWrites=true"


def test_mongo_cache_backend_config_repr_redacts_password():
    config = MongoCacheBackendConfig(
        uri="mongodb://user:password@example.com/cache",
        username="user",
        password="password",
    )

    assert "mongodb://user:password@" not in repr(config)
    assert "password@example.com" not in repr(config)
    assert "example.com" in repr(config)


def test_mongo_cache_config_redacts_password():
    config = MongoCacheConfig(
        uri="mongodb://user:password@example.com/cache",
        collection_name="requests",
        username="user",
        password="password",
    )

    assert "user:password@" not in repr(config)
    assert config.cache_stats_key == "mongodb://example.com/cache/requests"


def test_get_mongo_cache_backend_config_from_credentials_file(monkeypatch):
    monkeypatch.delenv("HELM_MONGODB_URI", raising=False)
    monkeypatch.delenv("HELM_MONGODB_USERNAME", raising=False)
    monkeypatch.delenv("HELM_MONGODB_PASSWORD", raising=False)
    with tempfile.TemporaryDirectory() as base_path:
        with open(os.path.join(base_path, "credentials.conf"), "w") as f:
            f.write(
                """
                mongoUri: "mongodb://example.com/cache"
                mongoUsername: "user"
                mongoPassword: "password"
                """
            )

        config = get_mongo_cache_backend_config(None, base_path)

    assert config == MongoCacheBackendConfig(
        uri="mongodb://example.com/cache",
        username="user",
        password="password",
    )


def test_get_mongo_cache_backend_config_requires_username_and_password(monkeypatch):
    monkeypatch.delenv("HELM_MONGODB_PASSWORD", raising=False)
    monkeypatch.setenv("HELM_MONGODB_URI", "mongodb://example.com/cache")
    monkeypatch.setenv("HELM_MONGODB_USERNAME", "user")

    with pytest.raises(ValueError, match="Set both HELM_MONGODB_USERNAME and HELM_MONGODB_PASSWORD"):
        get_mongo_cache_backend_config(None, "prod_env")
