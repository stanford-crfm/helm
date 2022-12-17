import os
from typing import Optional


class SlurmSharding:
    def __init__(self, num_shards: int):
        slurm_array_task_id: Optional[str] = os.environ.get("SLURM_ARRAY_TASK_ID")

        if slurm_array_task_id is None:
            raise ValueError("Expected SLURM_ARRAY_TASK_ID to be set; was the job submitted using sbatch --array?")
        if int(slurm_array_task_id) >= num_shards:
            raise ValueError(f"Expected SLURM_ARRAY_TASK_ID {slurm_array_task_id} to be < num_shard {num_shards}")

        self.shard_index: int = int(slurm_array_task_id)
        self.num_shards: int = num_shards

    def should_run(self, run_spec_name: str):
        return hash(run_spec_name) % self.num_shards == self.shard_index
