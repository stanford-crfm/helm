from typing import Dict, Optional, List
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec


class TokenizerSpec(ObjectSpec):
    pass


@dataclass(frozen=True)
class TokenizerConfig:
    """A model deployment is an accessible instance of this model (e.g. a hosted endpoint).

    A model can have model deployments."""

    name: str
    """Name of the model deployment."""

    end_of_text_token: str
    """End of text special token."""

    prefix_token: str
    """Prefix special token."""

    tokenizer_spec: TokenizerSpec
    """Specification for instantiating the client for this tokenizer."""


@dataclass(frozen=True)
class TokenizerConfigs:
    tokenizers: List[TokenizerConfig]


_name_to_tokenizer_config: Dict[str, TokenizerConfig] = {}


def register_tokenizer_configs_from_path(path: str) -> None:
    global _name_to_tokenizer_config
    hlog(f"Reading tokenizers from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    tokenizer_configs: TokenizerConfigs = cattrs.structure(raw, TokenizerConfigs)
    for tokenizer_config in tokenizer_configs.tokenizers:
        _name_to_tokenizer_config[tokenizer_config.name] = tokenizer_config


def get_tokenizer_config(name: str) -> Optional[TokenizerConfig]:
    return _name_to_tokenizer_config.get(name)
