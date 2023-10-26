import os
from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec


TOKENIEZR_CONFIGS_FILE = "tokenizer_configs.yaml"


class TokenizerSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for a tokenizer."""

    name: str
    """Name of the tokenizer."""

    tokenizer_spec: TokenizerSpec
    """Specification for instantiating the client for this tokenizer."""

    # TODO: Add `end_of_text_token`` and `prefix_token``


@dataclass(frozen=True)
class TokenizerConfigs:
    tokenizer_configs: List[TokenizerConfig]


_name_to_tokenizer_config: Dict[str, TokenizerConfig] = {}


def register_tokenizer_config(tokenizer_config: TokenizerConfig) -> None:
    _name_to_tokenizer_config[tokenizer_config.name] = tokenizer_config


def register_tokenizer_configs_from_path(path: str) -> None:
    hlog(f"Reading tokenizer configs from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    tokenizer_configs: TokenizerConfigs = cattrs.structure(raw, TokenizerConfigs)
    for tokenizer_config in tokenizer_configs.tokenizer_configs:
        register_tokenizer_config(tokenizer_config)


def maybe_register_tokenizer_configs_from_base_path(base_path: str) -> None:
    path = os.path.join(base_path, TOKENIEZR_CONFIGS_FILE)
    if os.path.exists(path):
        register_tokenizer_configs_from_path(path)


def get_tokenizer_config(name: str) -> Optional[TokenizerConfig]:
    return _name_to_tokenizer_config.get(name)
