from threading import Lock
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM
from helm.common.cache import CacheConfig

from helm.common.optional_dependencies import OptionalDependencyNotInstalled
from helm.clients.huggingface_client import HuggingFaceClient


_register_open_lm_lock = Lock()
_register_open_lm_done = False


def _register_open_lm_for_auto_model():
    """Register OpenLMForCausalLM for AutoModelForCausalLM."""
    try:
        from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
        from open_lm.utils.transformers.hf_config import OpenLMConfig
    except ModuleNotFoundError as e:
        # Provide manual instructions for installing open_lm from GitHub
        # because PyPI does not allow installing dependencies directly from GitHub.
        raise OptionalDependencyNotInstalled(
            f"Optional dependency {e.name} is not installed. "
            "Please run `pip install open_lm@git+https://github.com/mlfoundations/open_lm.git@main` to install it."
        ) from e

    with _register_open_lm_lock:
        global _register_open_lm_done
        if not _register_open_lm_done:
            AutoConfig.register("openlm", OpenLMConfig)
            AutoModelForCausalLM.register(OpenLMConfig, OpenLMforCausalLM)
        _register_open_lm_done = True


class OpenLMClient(HuggingFaceClient):
    """Client for OpenLM: https://github.com/mlfoundations/open_lm"""

    def __init__(self, cache_config: CacheConfig, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        _register_open_lm_for_auto_model()
        super().__init__(
            cache_config=cache_config, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
