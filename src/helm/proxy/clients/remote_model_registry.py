from typing import Dict, List, Optional
from dataclasses import dataclass
import re

from helm.common.hierarchical_logger import hlog
from helm.proxy.models import Model, MODEL_NAME_TO_MODEL, get_all_models
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.services.service import GeneralInfo


_remote_model_registry: Dict[str, Model] = {}


def get_remote_model(model_name: str) -> Optional[Model]:
    """Returns a General Model for the model_name."""
    return _remote_model_registry.get(model_name)


def check_and_register_remote_model(server_url: Optional[str], model_names: List[str]):
    if server_url is None:
        return
    try:
        service = RemoteService(server_url)
        info = service.get_general_info()
        models = {}
        for model in info.all_models:
            models[model.name] = model
        for model_name in model_names:
            if model_name in models:
                _remote_model_registry[model_name] = models[model_name]
            else:
                raise RuntimeError(f"remote service not contain {model_name}")
    except Exception as e:
        raise RuntimeError(f"check and register remote service error: {e}")
