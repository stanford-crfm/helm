from dataclasses import dataclass
from typing import Any, Dict, Optional
import dacite
import yaml


@dataclass
class RunnerConfigSpec:
    helm_max_concurrent_workers: int = -1
    slurm_monitor_interval: int = 60
    slurm_args: Optional[Dict[str, Any]] = None


RUNNER_CONFIG = RunnerConfigSpec()


def register_runner_config_from_path(dir_path: str) -> None:
    global RUNNER_CONFIG
    with open(dir_path, "r") as f:
        raw = yaml.safe_load(f)
    RUNNER_CONFIG = dacite.from_dict(RunnerConfigSpec, raw)
