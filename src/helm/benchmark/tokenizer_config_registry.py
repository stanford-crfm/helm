import os
from typing import Dict, Optional, List
from dataclasses import dataclass
import importlib_resources as resources

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.benchmark.model_metadata_registry import CONFIG_PACKAGE


TOKENIZER_CONFIGS_FILE: str = "tokenizer_configs.yaml"
TOKENIZERS_REGISTERED: bool = False


class TokenizerSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for a tokenizer."""

    name: str
    """Name of the tokenizer."""

    tokenizer_spec: TokenizerSpec
    """Specification for instantiating the client for this tokenizer."""

    end_of_text_token: Optional[str] = None
    """The end of text token."""

    prefix_token: Optional[str] = None
    """The prefix token."""


@dataclass(frozen=True)
class TokenizerConfigs:
    tokenizer_configs: List[TokenizerConfig]


ALL_TOKENIZER_CONFIGS: List[TokenizerConfig] = []
TOKENIZER_NAME_TO_CONFIG: Dict[str, TokenizerConfig] = {config.name: config for config in ALL_TOKENIZER_CONFIGS}


def register_tokenizer_config(tokenizer_config: TokenizerConfig) -> None:
    ALL_TOKENIZER_CONFIGS.append(tokenizer_config)
    TOKENIZER_NAME_TO_CONFIG[tokenizer_config.name] = tokenizer_config


def register_tokenizer_configs_from_path(path: str) -> None:
    global TOKENIZERS_REGISTERED
    hlog(f"Reading tokenizer configs from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    tokenizer_configs: TokenizerConfigs = cattrs.structure(raw, TokenizerConfigs)
    for tokenizer_config in tokenizer_configs.tokenizer_configs:
        register_tokenizer_config(tokenizer_config)
    TOKENIZERS_REGISTERED = True


def maybe_register_tokenizer_configs_from_base_path(path: str) -> None:
    """Register tokenizer configs from yaml file if the path exists."""
    if os.path.exists(path):
        register_tokenizer_configs_from_path(path)


def get_tokenizer_config(name: str) -> Optional[TokenizerConfig]:
    register_tokenizers_if_not_already_registered()
    return TOKENIZER_NAME_TO_CONFIG.get(name)


def register_tokenizers_if_not_already_registered() -> None:
    if not TOKENIZERS_REGISTERED:
        path: str = resources.files(CONFIG_PACKAGE).joinpath(TOKENIZER_CONFIGS_FILE)
        maybe_register_tokenizer_configs_from_base_path(path)
