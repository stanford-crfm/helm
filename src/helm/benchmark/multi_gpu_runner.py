import traceback
from typing import List
import os
import torch.multiprocessing as multiprocessing

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.runner import Runner, RunSpec, RunnerError

from helm.common.hierarchical_logger import hlog, htrack_block

from helm.benchmark.slurm_config_registry import SLURM_CONFIG

_MAX_CONCURRENT_WORKER_SLURM_JOBS_ENV_NAME = "HELM_MAX_CONCURRENT_WORKER_SLURM_JOBS"


def worker_initialize(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    hlog(f"Worker {gpu_id} initializing with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


class MultiGPURunner(Runner):
    """Runner that runs the entire benchmark on multiple GPUs.

    This is a thin wrapper around `Runner` that runs the entire benchmark on multiple GPUs using `multiprocessing`."""

    def __init__(
        self,
        execution_spec: ExecutionSpec,
        output_path: str,
        suite: str,
        skip_instances: bool,
        cache_instances: bool,
        cache_instances_only: bool,
        skip_completed_runs: bool,
        exit_on_error: bool,
    ):
        super().__init__(
            execution_spec=execution_spec,
            output_path=output_path,
            suite=suite,
            skip_instances=skip_instances,
            cache_instances=cache_instances,
            cache_instances_only=cache_instances_only,
            skip_completed_runs=skip_completed_runs,
            exit_on_error=exit_on_error,
        )
        # Configure max concurrent worker jobs from the environment variable.
        env_max_concurrent_worker_slurm_jobs = os.getenv(_MAX_CONCURRENT_WORKER_SLURM_JOBS_ENV_NAME)
        self.max_concurrent_worker_slurm_jobs = (
            int(env_max_concurrent_worker_slurm_jobs)
            if env_max_concurrent_worker_slurm_jobs
            else SLURM_CONFIG.helm_max_concurrent_worker_slurm_jobs
        )

    def safe_run_one(self, run_spec: RunSpec):
        register_builtin_configs_from_helm_package()
        if self.executor.execution_spec.local_path is not None:
            register_configs_from_directory(self.executor.execution_spec.local_path)

        try:
            with htrack_block(f"Running {run_spec.name}"):
                self.run_one(run_spec)
        except Exception as e:
            hlog(f"Error when running {run_spec.name}:\n{traceback.format_exc()}")
            return e

    def run_all(self, run_specs: List[RunSpec]):
        """Run the entire benchmark on multiple GPU"""

        # Set the start method to forkserver to avoid issues with CUDA.
        multiprocessing.set_start_method("forkserver")

        with multiprocessing.Pool(processes=self.max_concurrent_worker_slurm_jobs) as pool:
            # Pin GPUs to each worker process
            pool.starmap(worker_initialize, [(i,) for i in range(self.max_concurrent_worker_slurm_jobs)])

            # Run all queued tasks
            error_msgs = pool.map(self.safe_run_one, run_specs)

        # Raise exception for failed runs, if any.
        failed_run_names = [
            run_spec.name for error_msg, run_spec in zip(error_msgs, run_specs) if error_msg is not None
        ]
        if failed_run_names:
            failed_runs_str = ", ".join([f'"{run_name}"' for run_name in failed_run_names])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")
