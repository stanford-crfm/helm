import signal
import threading
import traceback
from typing import List
import os
import time
import torch
import torch.multiprocessing as multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.runner import Runner, RunSpec, RunnerError
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.benchmark.runner_config_registry import RUNNER_CONFIG

_MAX_CONCURRENT_WORKERS_ENV_NAME = "HELM_MAX_CONCURRENT_WORKERS"


# From
# https://stackoverflow.com/questions/71300294/how-to-terminate-pythons-processpoolexecutor-when-parent-process-dies
def start_thread_to_terminate_when_parent_process_dies(ppid):
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def initialize_worker(gpu_id: int):
    hlog(f"Worker {gpu_id} initializing")

    # Wait for 0.1 seconds to ensure all workers are initialized with different CUDA_VISIBLE_DEVICES
    time.sleep(0.1)

    # Pin GPU to worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Necessary for code_metrics in humaneval to work properly
    multiprocessing.set_start_method("fork", force=True)


class MultiGPURunner(Runner):
    """Runner that runs the entire benchmark on multiple GPUs.

    This is a thin wrapper around `Runner` that runs the entire benchmark on
    multiple GPUs using `multiprocessing`.

    Note that this runner will load multiple models into memory at the same
    time if your running configuration specifies that, similar to the `Runner`
    class. `SlurmRunner` on the other hand will load at most one model on a
    GPU"""

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
        env_max_concurrent_workers = os.getenv(_MAX_CONCURRENT_WORKERS_ENV_NAME)
        self.max_concurrent_workers = (
            int(env_max_concurrent_workers)
            if env_max_concurrent_workers
            else (
                RUNNER_CONFIG.helm_max_concurrent_workers
                if RUNNER_CONFIG.helm_max_concurrent_workers > 0
                else torch.cuda.device_count()
            )
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

        with Pool(
            max_workers=self.max_concurrent_workers,
            initializer=start_thread_to_terminate_when_parent_process_dies,
            initargs=(os.getpid(),),
        ) as pool:
            # Pin GPUs to each worker process
            pool.map(initialize_worker, [i for i in range(self.max_concurrent_workers)])

            # Run all queued tasks
            error_msgs = list(tqdm(pool.map(self.safe_run_one, run_specs), total=len(run_specs), disable=None))

        # Raise exception for failed runs, if any.
        failed_run_names = [
            run_spec.name for error_msg, run_spec in zip(error_msgs, run_specs) if error_msg is not None
        ]
        if failed_run_names:
            failed_runs_str = ", ".join([f'"{run_name}"' for run_name in failed_run_names])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")
