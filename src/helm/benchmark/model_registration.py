from typing import Any, Dict, Iterable
from dataclasses import dataclass
from helm.common.general import parse_hocon
from helm.proxy.clients.ai21_model_registry import AI21ModelConfig, register_ai21_model_config

from cattrs import structure

@dataclass(frozen=True)
class ModelConfig:
    client_type: str
    config: Dict[str, Any]

def register_model_configs_from_path(path: str) -> ModelConfig:
    with open(path, "r") as f:
        raw = parse_hocon(f.read())
        name_to_config = structure(raw, Dict[str, ModelConfig])
        # print(raw)
        for name, config in name_to_config.items():
            if config.client_type == "ai21":
                # print("[debug:yifanmai] registering config")
                register_ai21_model_config(name, AI21ModelConfig(url=config.config["url"], api_key=config.config.get("api_key")))
            else:
                raise Exception(f"Unknown client type {config.client_type}")


# def read_model_configs_from_paths(paths: Iterable[str]) -> ModelConfig:
#     """Read a HOCON file `path` and return the `RunEntry`s."""
#     run_entries = ModelConfig([])
#     for path in paths:
#         with open(path) as f:
#             raw = parse_hocon(f.read())
