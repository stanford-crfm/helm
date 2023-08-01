from typing import Dict, Optional, Union
from dataclasses import dataclass
import re
import os
from helm.common.hierarchical_logger import hlog
from helm.proxy.models import (
    AI21_TOKENIZER_TAG,
    Model,
    ALL_MODELS,
    MODEL_NAME_TO_MODEL,
    TEXT_MODEL_TAG,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    get_model_names_with_tag,
)

_COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/{model_engine}/complete"
_EXPERIMENTAL_COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/experimental/{model_engine}/complete"
_AI21_PREFIX = "ai21/"

@dataclass(frozen=True)
class AI21ModelConfig:
    url: str
    api_key: Optional[str] = None

def _get_default_ai21_model_url(model_name: str):
    assert model_name.startswith(_AI21_PREFIX)
    model_engine = model_name[len(_AI21_PREFIX):]
    url_template: str = (
        _EXPERIMENTAL_COMPLETION_URL_TEMPLATE
        if model_engine == "j1-grande-v2-beta"
        else _COMPLETION_URL_TEMPLATE
    )
    return url_template.format(model_engine=model_engine)


# Initialize registry with local models from models.py
_ai21_model_registry: Dict[str, AI21ModelConfig] = {
    name: AI21ModelConfig(url=_get_default_ai21_model_url(name))
    for name in get_model_names_with_tag(AI21_TOKENIZER_TAG)
}


def register_ai21_model_config(name: str, config: AI21ModelConfig) -> None:
    if not name.startswith("ai21/"):
        raise Exception("AI21 model names must start with ai21/")
    if name in _ai21_model_registry:
        raise Exception(f"There is already an AI21 model registered for name {name}")
    _ai21_model_registry[name] = config
    model = Model(
        group="ai21",
        name=name,
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, AI21_TOKENIZER_TAG],
    )
    MODEL_NAME_TO_MODEL[name] = model
    ALL_MODELS.append(model)
    hlog(f"Registered AI21 model: {model} config: {config}")
    return config


def get_ai21_model_config(model_name: str) -> Optional[AI21ModelConfig]:
    # print(f"[debug:yifanmai] registry: {_ai21_model_registry}")
    # import pprint
    # pprint.pprint(_ai21_model_registry)
    return _ai21_model_registry.get(model_name)
