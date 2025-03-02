import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import argparse
import os
import time
import shlex
import sys

from helm.common.codec import from_json, to_json
from helm.common.general import write
from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.runner import Runner, RunSpec, RunnerError
from helm.benchmark.slurm_jobs import (
    submit_slurm_job,
    cancel_slurm_job,
    get_slurm_job_state,
    SlurmJobState,
    ACTIVE_SLURM_JOB_STATES,
    TERMINAL_SLURM_JOB_STATES,
    FAILURE_SLURM_JOB_STATES,
)
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack_block

from helm.benchmark.runner_config_registry import RUNNER_CONFIG

_MAX_CONCURRENT_WORKER_SLURM_JOBS_ENV_NAME = "HELM_MAX_CONCURRENT_WORKER_SLURM_JOBS"
_SLURM_NODE_NAMES_ENV_NAME = "HELM_SLURM_NODE_NAMES"
_DEFAULT_MAX_CONCURRENT_WORKER_SLURM = 4


@dataclass
class SlurmRunnerSpec:
    """Arguments to instantiate a SlurmRunner."""

    execution_spec: ExecutionSpec
    output_path: str
    suite: str
    skip_instances: bool
    cache_instances: bool
    cache_instances_only: bool
    skip_completed_runs: bool
    exit_on_error: bool

    def to_kwargs(self):
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


@dataclass
class _SlurmJobInfo:
    """Slurm job information tracked by Slurm Runner."""

    id: int
    state: str


class SlurmRunner(Runner):
    """Runner that runs the entire benchmark on Slurm.

    See the documentation of `run_all()` for more details."""

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
        self.slurm_runner_spec = SlurmRunnerSpec(
            execution_spec=execution_spec,
            output_path=output_path,
            suite=suite,
            skip_instances=skip_instances,
            cache_instances=cache_instances,
            cache_instances_only=cache_instances_only,
            skip_completed_runs=skip_completed_runs,
            exit_on_error=exit_on_error,
        )
        # Extra validation: Check that SlurmRunnerSpec can be used to initialize Runner.
        super().__init__(**self.slurm_runner_spec.to_kwargs())
        self.slurm_base_dir = os.path.join("slurm", datetime.now().isoformat(timespec="seconds"))
        self.run_specs_dir = os.path.join(self.slurm_base_dir, "run_specs")
        self.logs_dir = os.path.join(self.slurm_base_dir, "logs")
        self.slurm_runner_spec_path = os.path.join(self.slurm_base_dir, "slurm_runner_spec.json")

        # Configure max concurrent worker Slurm jobs from the environment variable.
        env_max_concurrent_worker_slurm_jobs = os.getenv(_MAX_CONCURRENT_WORKER_SLURM_JOBS_ENV_NAME)
        self.max_concurrent_worker_slurm_jobs = (
            int(env_max_concurrent_worker_slurm_jobs)
            if env_max_concurrent_worker_slurm_jobs
            else (
                RUNNER_CONFIG.helm_max_concurrent_workers
                if RUNNER_CONFIG.helm_max_concurrent_workers > 0
                else _DEFAULT_MAX_CONCURRENT_WORKER_SLURM
            )
        )

    def run_all(self, run_specs: List[RunSpec]):
        """Run the entire benchmark on Slurm, where each RunSpec is run in its own Slurm job.

        This process functions as a manager job that does the following:

        1. Write SlurmRunnerSpec to a file.
        2. For each RunSpec:
            a. Write the RunSpec to a file.
            b. Submit a worker Slurm job via sbatch.
            c. The worker Slurm job will read the SlurmRunnerSpec and RunSpec from the files
               and run the RunSpec.
        3. Monitor all the worker jobs:
            a. If skip_completed_runs is True, terminate and cancel all the worker Slurm jobs
               if any worker Slurm job fails.
            b. If skip_completed_runs is False, terminate when all worker Slurm jobs are completed.
        4. If the manager job has terminated or encounters an exception,
           cancel all the worker jobs upon exiting. (best-effort)
        """
        # This directory will be used for coordinating with workers
        ensure_directory_exists(self.slurm_base_dir)
        ensure_directory_exists(self.run_specs_dir)
        ensure_directory_exists(self.logs_dir)

        # Write the SlurmRunnerSpec to a file
        slurm_runner_spec_json = to_json(self.slurm_runner_spec)
        slurm_runner_spec_path = os.path.join(self.slurm_base_dir, "slurm_runner_spec.json")
        hlog(f"Writing SlurmRunnerSpec to {slurm_runner_spec_path}")
        write(file_path=slurm_runner_spec_path, content=slurm_runner_spec_json)

        skipped_run_specs: List[RunSpec] = []
        queued_run_specs: List[RunSpec] = []
        # When running with multiple models, sorting by RunSpec.name is a heuristic that tries to
        # spread out the load evenly across multiple models, in order to avoid overloading any single model.
        for run_spec in sorted(run_specs, key=lambda run_spec: run_spec.name):
            if self.skip_completed_runs and self._is_run_completed(self._get_run_path(run_spec)):
                skipped_run_specs.append(run_spec)
            else:
                queued_run_specs.append(run_spec)
        # Reverse queued runs because we pop from the end of the list.
        queued_run_specs.reverse()

        skipped_runs_json = to_json([run_spec.name for run_spec in skipped_run_specs])
        if skipped_run_specs:
            hlog("Skipped completed runs because --skip-completed-runs was set: " f"{skipped_runs_json}")
        skipped_runs_path = os.path.join(self.slurm_base_dir, "skipped_runs.json")
        # Write skipped run specs file anyway even if empty.
        # This makes things more convenient for downstream status monitoring tools.
        hlog(f"Writing skipped runs to {skipped_runs_path}")
        write(file_path=skipped_runs_path, content=skipped_runs_json)

        # Info for all worker Slurm jobs
        run_name_to_slurm_job_info: Dict[str, _SlurmJobInfo] = {}

        # Location to persist the info for all worker Slurm jobs
        worker_slurm_jobs_path = os.path.join(self.slurm_base_dir, "worker_slurm_jobs.json")

        # Callback for cleaning up worker Slurm jobs
        def cancel_all_jobs():
            """Cancels all submitted worker Slurm jobs that are in a non-terminal state."""
            with htrack_block("Cleaning up by cancelling all worker Slurm jobs"):
                # TODO: Cancel multiple jobs in a single call to Slurm
                for run_name, slurm_job_info in run_name_to_slurm_job_info.items():
                    if slurm_job_info.state not in TERMINAL_SLURM_JOB_STATES:
                        hlog(f"Cancelling worker Slurm job run {run_name} with Slurm job ID {slurm_job_info.id}")
                        cancel_slurm_job(slurm_job_info.id)
                        slurm_job_info.state = SlurmJobState.CANCELLED
            run_name_to_slurm_job_info_json = to_json(run_name_to_slurm_job_info)
            hlog(f"Worker Slurm jobs: {run_name_to_slurm_job_info_json}")
            hlog(f"Writing worker Slurm job states to {worker_slurm_jobs_path}")
            write(file_path=worker_slurm_jobs_path, content=run_name_to_slurm_job_info_json)

        try:
            # Monitor submitted Slurm jobs for RunSpecs until an exit condition is triggered.
            with htrack_block("Managing worker Slurm jobs"):
                while True:
                    num_active_slurm_jobs = len(
                        [
                            slurm_job_info
                            for slurm_job_info in run_name_to_slurm_job_info.values()
                            if slurm_job_info.state in ACTIVE_SLURM_JOB_STATES
                        ]
                    )
                    available_concurrency = self.max_concurrent_worker_slurm_jobs - num_active_slurm_jobs
                    while available_concurrency > 0 and queued_run_specs:
                        available_concurrency -= 1
                        run_spec = queued_run_specs.pop()
                        hlog(f"Submitting Slurm job for {run_spec.name}")
                        slurm_job_id = self._submit_slurm_job_for_run_spec(run_spec)
                        run_name_to_slurm_job_info[run_spec.name] = _SlurmJobInfo(
                            id=slurm_job_id, state=SlurmJobState.PENDING
                        )
                    queued_runs_json = to_json([run_spec.name for run_spec in queued_run_specs])
                    queued_runs_path = os.path.join(self.slurm_base_dir, "queued_runs.json")
                    hlog(f"Writing queued runs to {queued_runs_path}")
                    write(file_path=queued_runs_path, content=queued_runs_json)

                    hlog("Fetching states of worker Slurm jobs from Slurm")
                    # TODO: Get the states of multiple jobs in a single call to Slurm
                    for slurm_job_info in run_name_to_slurm_job_info.values():
                        if slurm_job_info.state not in TERMINAL_SLURM_JOB_STATES:
                            slurm_job_info.state = get_slurm_job_state(slurm_job_info.id)
                    run_name_to_slurm_job_info_json = to_json(run_name_to_slurm_job_info)
                    hlog(f"Worker Slurm jobs: {run_name_to_slurm_job_info_json}")
                    hlog(f"Writing worker Slurm job states to {worker_slurm_jobs_path}")
                    write(file_path=worker_slurm_jobs_path, content=run_name_to_slurm_job_info_json)

                    # Check termination conditions
                    if self.exit_on_error and any(
                        [
                            slurm_job_info.state in FAILURE_SLURM_JOB_STATES
                            for slurm_job_info in run_name_to_slurm_job_info.values()
                        ]
                    ):
                        hlog("Some worker Slurm job failed and --exit_on_error was set.")
                        break
                    if not queued_run_specs and all(
                        [
                            slurm_job_info.state in TERMINAL_SLURM_JOB_STATES
                            for slurm_job_info in run_name_to_slurm_job_info.values()
                        ]
                    ):
                        hlog("All worker Slurm jobs completed.")
                        break

                    # Refresh every minute
                    time.sleep(RUNNER_CONFIG.slurm_monitor_interval)
        finally:
            # Cleanup by cancelling all jobs during program termination or if an exception is raised.
            cancel_all_jobs()

        # Raise exception for failed runs, if any.
        failed_run_names = [
            run_name
            for run_name, slurm_job_info in run_name_to_slurm_job_info.items()
            if slurm_job_info.state in FAILURE_SLURM_JOB_STATES
        ]
        if failed_run_names:
            failed_runs_str = ", ".join([f'"{run_name}"' for run_name in failed_run_names])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")

    def _submit_slurm_job_for_run_spec(self, run_spec: RunSpec) -> int:
        """Create a Slurm job for the RunSpec and return the Slurm job ID."""
        # Create a worker Slurm job that reads from the SlurmRunnerSpec and RunSpec files
        run_name = run_spec.name
        run_spec_json = to_json(run_spec)
        run_spec_path = os.path.join(self.run_specs_dir, f"{run_name}.json")
        hlog(f"Writing RunSpec for run {run_name} to {run_spec_path}")
        write(file_path=run_spec_path, content=run_spec_json)

        log_path = os.path.join(self.logs_dir, f"{run_name}.log")
        # Requires that SlurmRunnerSpec has already been written to self.slurm_runner_spec_path.
        # It should have been written at the start of self.run_all()
        command = shlex.join(
            [
                sys.executable,
                "-m",
                SlurmRunner.__module__,
                "--slurm-runner-spec-path",
                self.slurm_runner_spec_path,
                "--run-spec-path",
                run_spec_path,
            ]
        )
        if RUNNER_CONFIG.slurm_args is None:
            raw_slurm_args: Dict[str, str] = {
                "account": "nlp",
                "cpus_per_task": "4",
                "mem": "32G",
                "gres": "gpu:0",
                "open_mode": "append",
                "partition": "john",
                "time": "14-0",  # Deadline of 14 days
                "mail_type": "FAIL",
                "job_name": run_name,
                "output": log_path,
                "chdir": os.getcwd(),
            }
            # TODO: Move resource requirements into RunSpec.
            slurm_node_names = os.getenv(_SLURM_NODE_NAMES_ENV_NAME)
            if run_spec.name.startswith("msmarco:"):
                raw_slurm_args["mem"] = "64G"
            if "device=cuda" in run_spec.name:
                raw_slurm_args["gres"] = "gpu:1"
                raw_slurm_args["partition"] = "jag-hi"
            if "model=huggingface" in run_spec.name:
                raw_slurm_args["gres"] = "gpu:1"
                raw_slurm_args["partition"] = "sphinx"
                if not slurm_node_names or "sphinx" not in slurm_node_names:
                    raise Exception(
                        f"Environment variable {_SLURM_NODE_NAMES_ENV_NAME} must be set to sphinx node names"
                    )
            if slurm_node_names:
                raw_slurm_args["nodelist"] = slurm_node_names

        else:
            raw_slurm_args = RUNNER_CONFIG.slurm_args

            dynamic_slurm_args = {
                "job_name": run_name,
                "output": log_path,
                "chdir": os.getcwd(),
            }

            # User should not set these manually, overwrite them if necessary
            raw_slurm_args.update(dynamic_slurm_args)

        slurm_args: Dict[str, str] = {key: shlex.quote(value) for key, value in raw_slurm_args.items()}
        # Uncomment this to get notification emails from Slurm for Slurm worker jobs.
        # slurm.set_mail_user(os.getenv("USER"))
        hlog(f"Submitting worker Slurm job for run {run_name} with command: {command}")
        time.sleep(0.1)  # Delay to avoid overwhelming Slurm
        slurm_job_id = submit_slurm_job(command, slurm_args)
        hlog(f"Worker Slurm job submitted for run {run_name} with Slurm job ID: {slurm_job_id}")
        return slurm_job_id


def main():
    """Entry point for the SlurmRunner's worker Slurm jobs that run a single RunSpec.

    This entry point should only be used by SlurmRunner. Users should use `helm-run` instead.
    SlurmRunner has to use this entry point instead of helm-run because there is no way to
    specify the worker Slurm job parameters through `helm-run`. In particular, there is no way
    to run a specific `RunSpec` using the `--run-entries` parameter of `helm-run`, because the
    `run-entries` argument contains `RunEntry` description (not `RunSpec`s), and there is no way to
    convert a `RunSpec` into a `RunEntry` description."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurm-runner-spec-path",
        type=str,
        help="Path to the SlurmRunnerSpec JSON file",
        required=True,
    )
    parser.add_argument(
        "--run-spec-path",
        type=str,
        help="Path to the RunSpec JSON file",
        required=True,
    )
    args = parser.parse_args()

    # Deserialize SlurmRunner and RunSpec from the given files, then run the RunSpec with the SlurmRunner.
    with open(args.slurm_runner_spec_path, "r") as f:
        slurm_runner_spec = from_json(f.read(), SlurmRunnerSpec)
    with open(args.run_spec_path, "r") as f:
        run_spec = from_json(f.read(), RunSpec)

    register_builtin_configs_from_helm_package()
    if slurm_runner_spec.execution_spec.local_path is not None:
        register_configs_from_directory(slurm_runner_spec.execution_spec.local_path)

    slurm_runner = SlurmRunner(**slurm_runner_spec.to_kwargs())
    slurm_runner.run_one(run_spec)


if __name__ == "__main__":
    main()
