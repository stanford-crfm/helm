import threading
from typing import Callable, Dict, Optional, List, TypeVar
from dataclasses import dataclass

import cattrs
import yaml

from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec
from helm.common.plugins import import_all_modules_in_package


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
    hlog(f"Reading tokenizer configs from {path}...")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    tokenizer_configs: TokenizerConfigs = cattrs.structure(raw, TokenizerConfigs)
    for tokenizer_config in tokenizer_configs.tokenizer_configs:
        register_tokenizer_config(tokenizer_config)


_DISCOVER_TOKENIZER_CONFIG_GENERATORS_LOCK = threading.Lock()
_DISCOVER_TOKENIZER_CONFIG_GENERATORS_DONE = False


def discover_tokenizer_config_generators() -> None:
    """Discover and register all tokenizer config generators in helm.benchmark.tokenizer_configs"""
    global _DISCOVER_TOKENIZER_CONFIG_GENERATORS_DONE
    with _DISCOVER_TOKENIZER_CONFIG_GENERATORS_LOCK:
        if _DISCOVER_TOKENIZER_CONFIG_GENERATORS_DONE:
            return
        _DISCOVER_TOKENIZER_CONFIG_GENERATORS_DONE = True

        import helm.benchmark.tokenizer_configs  # noqa

        import_all_modules_in_package(helm.benchmark.tokenizer_configs)


def get_tokenizer_config(name: str) -> Optional[TokenizerConfig]:
    tokenizer_config: Optional[TokenizerConfig] = TOKENIZER_NAME_TO_CONFIG.get(name)
    if not tokenizer_config:
        discover_tokenizer_config_generators()
        for prefix, tokenizer_config_generator in _REGISTERED_TOKENIZER_CONFIG_GENERATORS.items():
            if name.startswith(prefix):
                tokenizer_config = tokenizer_config_generator(name)

    return tokenizer_config


TokenizerConfigGenerator = Callable[[str], TokenizerConfig]
"""A function that takes in a model name and returns a TokenizerConfig"""


_REGISTERED_TOKENIZER_CONFIG_GENERATORS: Dict[str, TokenizerConfigGenerator] = {}
"""Dict of prefixes to TokenizerConfigGenerators."""


F = TypeVar("F", bound=TokenizerConfigGenerator)


def tokenizer_config_generator(prefix: str) -> Callable[[F], F]:
    """Register the TokenizerConfigGenerator for the given model name prefix."""

    def wrap(func: F) -> F:
        key = prefix.strip("/") + "/"
        if key in _REGISTERED_TOKENIZER_CONFIG_GENERATORS:
            raise ValueError(f"A TokenizerConfigGenerator with prefix {key} already exists")
        _REGISTERED_TOKENIZER_CONFIG_GENERATORS[key] = func
        return func

    return wrap
