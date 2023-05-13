from typing import Dict, Optional
from dataclasses import dataclass
import re
from helm.common.hierarchical_logger import hlog
from helm.proxy.models import (
    Model,
    ALL_MODELS,
    MODEL_NAME_TO_MODEL,
    TEXT_MODEL_TAG,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
)


@dataclass(frozen=True)
class HuggingFaceModelConfig:
    namespace: Optional[str]
    """Name of the group or user that owns the model. e.g. 'stanford-crfm'

    May be None if the model (e.g. gpt2) does not have a namespace."""

    model_name: str
    """Name of the model. e.g. 'BioMedLM'

    Does not include the namespace."""

    revision: Optional[str]
    """Revision of the model to use e.g. 'main'.

    If None, use the default revision."""

    @property
    def model_id(self) -> str:
        """Return the model ID.

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'"""
        if self.namespace:
            return f"{self.namespace}/{self.model_name}"
        return self.model_name

    def __str__(self) -> str:
        """Return the full model name used by HELM in the format "[namespace/]model_name[@revision]".

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'
        - 'stanford-crfm/BioMedLM@main'"""
        result = self.model_name
        if self.namespace:
            result = f"{self.namespace}/{result}"
        if self.revision:
            result = f"{result}@{self.revision}"
        return result

    @staticmethod
    def from_string(raw: str) -> "HuggingFaceModelConfig":
        """Parses a string in the format "[namespace/]model_name[@revision]" to a HuggingFaceModelConfig.

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'
        - 'stanford-crfm/BioMedLM@main'"""
        pattern = r"((?P<namespace>[^/@]+)/)?(?P<model_name>[^/@]+)(@(?P<revision>[^/@]+))?"
        match = re.fullmatch(pattern, raw)
        if not match:
            raise ValueError(f"Could not parse model name: '{raw}'; Expected format: [namespace/]model_name[@revision]")
        model_name = match.group("model_name")
        assert model_name
        return HuggingFaceModelConfig(
            namespace=match.group("namespace"), model_name=model_name, revision=match.group("revision")
        )


_huggingface_model_registry: Dict[str, HuggingFaceModelConfig] = {}


def register_huggingface_model_config(model_name: str) -> HuggingFaceModelConfig:
    """Register a AutoModelForCausalLM model from Hugging Face Model Hub for later use.

    model_name format: namespace/model_name[@revision]"""
    config = HuggingFaceModelConfig.from_string(model_name)
    if config.model_id in _huggingface_model_registry:
        raise ValueError(f"A Hugging Face model is already registered for model_id {model_name}")
    _huggingface_model_registry[model_name] = config

    # HELM model names require a namespace
    if not config.namespace:
        raise Exception("Registration of Hugging Face models without a namespace is not supported")
    if model_name in MODEL_NAME_TO_MODEL:
        raise ValueError(f"A HELM model is already registered for model name: {model_name}")
    description = f"HuggingFace model {config.model_id}"
    if config.revision:
        description += f" at revision {config.revision}"
    model = Model(
        group=config.namespace,
        name=model_name,
        display_name=model_name,
        creator_organization=config.namespace,
        description=description,
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    )
    MODEL_NAME_TO_MODEL[model_name] = model
    ALL_MODELS.append(model)
    hlog(f"Registered Hugging Face model: {model} config: {config}")
    return config


def get_huggingface_model_config(model_name: str) -> Optional[HuggingFaceModelConfig]:
    """Returns a HuggingFaceModelConfig for the model_id."""
    return _huggingface_model_registry.get(model_name)
