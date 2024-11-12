import os
from typing import Optional, Dict, Union

from helm.benchmark.model_deployment_registry import (
    ClientSpec,
    ModelDeployment,
    register_model_deployment,
)
from helm.benchmark.model_metadata_registry import (
    get_model_metadata,
    get_unknown_model_metadata,
    register_model_metadata,
)
from helm.benchmark.tokenizer_config_registry import TokenizerConfig, TokenizerSpec, register_tokenizer_config
from helm.common.hierarchical_logger import hlog
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer


def register_huggingface_model(
    helm_model_name: str,
    pretrained_model_name_or_path: str,
    revision: Optional[str] = None,
) -> None:
    object_spec_args: Dict[str, Union[str, bool]] = {"pretrained_model_name_or_path": pretrained_model_name_or_path}
    if revision:
        object_spec_args["revision"] = revision

    # Auto-infer model properties from the tokenizer.
    create_tokenizer_args: Dict[str, str] = {"pretrained_model_name_or_path": pretrained_model_name_or_path}
    if revision:
        create_tokenizer_args["revision"] = revision
    with HuggingFaceTokenizer.create_tokenizer(**create_tokenizer_args) as tokenizer:
        max_sequence_length = tokenizer.model_max_length
        end_of_text_token = tokenizer.eos_token or ""
        prefix_token = tokenizer.bos_token or ""
    # If the tokenizer config has a model_max_length of 1000000000000000019884624838656
    # it means that model creator did not specify model_max_length.
    if max_sequence_length > 1_000_000:
        raise ValueError(
            f"Could not infer the model_max_length of Hugging Face model {pretrained_model_name_or_path}, so "
            f"--enable-huggingface-models and --enable-local-huggingface-models cannot be used for this model. "
            f"Please configure the model using prod_env/model_deployments.yaml instead."
        )

    model_deployment = ModelDeployment(
        name=helm_model_name,
        client_spec=ClientSpec(
            class_name="helm.clients.huggingface_client.HuggingFaceClient",
            args=object_spec_args,
        ),
        model_name=helm_model_name,
        tokenizer_name=helm_model_name,
        max_sequence_length=max_sequence_length,
    )

    # We check if the model is already registered because we don't want to
    # overwrite the model metadata if it's already registered.
    # If it's not registered, we register it, as otherwise an error would be thrown
    # when we try to register the model deployment.
    try:
        _ = get_model_metadata(model_name=helm_model_name)
    except ValueError:
        register_model_metadata(get_unknown_model_metadata(helm_model_name))
        hlog(f"Registered default metadata for model {helm_model_name}")

    register_model_deployment(model_deployment)
    tokenizer_config = TokenizerConfig(
        name=helm_model_name,
        tokenizer_spec=TokenizerSpec(
            class_name="helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
            args=object_spec_args,
        ),
        end_of_text_token=end_of_text_token,
        prefix_token=prefix_token,
    )
    register_tokenizer_config(tokenizer_config)


def register_huggingface_hub_model_from_flag_value(raw_model_string: str) -> None:
    raw_model_string_parts = raw_model_string.split("@")
    pretrained_model_name_or_path: str
    revision: Optional[str]
    if len(raw_model_string_parts) == 1:
        pretrained_model_name_or_path, revision = raw_model_string_parts[0], None
    elif len(raw_model_string_parts) == 2:
        pretrained_model_name_or_path, revision = raw_model_string_parts
    else:
        raise ValueError(
            f"Could not parse Hugging Face flag value: '{raw_model_string}'; "
            "Expected format: namespace/model_engine[@revision]"
        )
    register_huggingface_model(
        helm_model_name=raw_model_string,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
    )


def register_huggingface_local_model_from_flag_value(path: str) -> None:
    if not path:
        raise ValueError("Path to Hugging Face model must be non-empty")
    path_parts = os.path.split(path)
    helm_model_name = f"huggingface/{path_parts[-1]}"
    register_huggingface_model(
        helm_model_name=helm_model_name,
        pretrained_model_name_or_path=path,
    )
