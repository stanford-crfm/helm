"""Temporary test for preserving invariants during the model / tokenizer / window service refactor.

Delete this after the refactor is done."""

from typing import Optional

import pytest
from tempfile import TemporaryDirectory
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_deployment_registry import (
    get_model_deployment,
    ModelDeployment,
    ALL_MODEL_DEPLOYMENTS,
)
from helm.benchmark.model_metadata_registry import get_model_metadata, ModelMetadata
from helm.benchmark.tokenizer_config_registry import TokenizerConfig, get_tokenizer_config
from helm.benchmark.window_services.test_utils import get_tokenizer_service
from helm.proxy.clients.client import Client
from helm.proxy.tokenizers.tokenizer import Tokenizer
from helm.benchmark.window_services.window_service import WindowService

from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.proxy.clients.auto_client import AutoClient
from helm.proxy.tokenizers.auto_tokenizer import AutoTokenizer


# HACK: This looks like it should be done in a setup_class()
# for the test below but apparently pytest first check the parametrize
# before running the setup_class().
# Therefore ALL_MODEL_DEPLOYMENTS is empty and no test would be run,
# so we need to do this here.
register_builtin_configs_from_helm_package()

INT_MAX: int = 2**31 - 1


class TestModelProperties:
    @pytest.mark.parametrize("deployment_name", [deployment.name for deployment in ALL_MODEL_DEPLOYMENTS])
    def test_models_has_window_service(self, deployment_name: str):
        with TemporaryDirectory() as tmpdir:
            auto_client = AutoClient({}, tmpdir, "")
            auto_tokenizer = AutoTokenizer({}, tmpdir, "")
            tokenizer_service = get_tokenizer_service(tmpdir)

            # Loading the TokenizerConfig and ModelMetadat ensures that they are valid.
            deployment: ModelDeployment = get_model_deployment(deployment_name)
            tokenizer_name: str = deployment.tokenizer_name if deployment.tokenizer_name else deployment_name
            tokenizer_config: Optional[TokenizerConfig] = get_tokenizer_config(tokenizer_name)
            assert tokenizer_config is not None
            model: ModelMetadata = get_model_metadata(
                deployment.model_name if deployment.model_name else deployment_name
            )

            # Can't test lit-gpt client because it requires manual dependencies
            if "lit-gpt" in model.name:
                return

            # Can't test Llama 2 because it requires Hugging Face credentials
            if "llama-2-" in model.name:
                return

            # Can't test Vertex AI because it requires Google credentials
            if "text-bison" in model.name or "text-unicorn" in model.name:
                return

            # Loads the model, window service and tokenizer
            # which checks that the model, window service and tokenizer are all valid,
            # and that no Client, WindowService or Tokenizer are crashing.
            client: Client = auto_client._get_client(deployment_name)  # noqa: F841
            window_service: WindowService = WindowServiceFactory.get_window_service(deployment_name, tokenizer_service)
            tokenizer: Tokenizer = auto_tokenizer._get_tokenizer(tokenizer_name)  # noqa: F841

            # Verify that the parameters that are redundant between the ModelDeployment, Tokenizer and the
            # WindowService are the same.
            assert window_service.tokenizer_name == deployment.tokenizer_name
            assert window_service.max_sequence_length == deployment.max_sequence_length
            assert (
                window_service.max_request_length == deployment.max_request_length
                if deployment.max_request_length
                else deployment.max_sequence_length
            )
            assert (
                window_service.max_sequence_and_generated_tokens_length
                == deployment.max_sequence_and_generated_tokens_length
                if deployment.max_sequence_and_generated_tokens_length
                else INT_MAX
            )
            assert tokenizer_config.end_of_text_token == window_service.end_of_text_token
            assert tokenizer_config.prefix_token == window_service.prefix_token

            # TODO: Add a dummy tokenize, decode and make_request request to each client/tokenizer
            # Do this once we have a proper Cache for tests.
