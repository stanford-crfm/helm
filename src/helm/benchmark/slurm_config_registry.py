from dataclasses import dataclass
from typing import Any, Dict, Optional
import dacite
import yaml


@dataclass
class SlurmConfigSpec:
    helm_max_concurrent_worker_slurm_jobs: int = 8
    slurm_monitor_interval: int = 60
    slurm_args: Optional[Dict[str, Any]] = None


SLURM_CONFIG = SlurmConfigSpec()


def register_slurm_config_from_path(dir_path: str) -> None:
    global SLURM_CONFIG
    with open(dir_path, "r") as f:
        raw = yaml.safe_load(f)
    SLURM_CONFIG = dacite.from_dict(SlurmConfigSpec, raw)
