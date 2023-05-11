import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import argparse
import json
import os
import sys

import dacite

from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.runner import Runner, RunSpec, RunnerError
from helm.benchmark.slurm_jobs import (
    submit_slurm_job,
    cancel_slurm_job,
    get_slurm_job_state,
    SlurmJobState,
    TERMINAL_SLURM_JOB_STATES,
    FAILURE_SLURM_JOB_STATES,
)
from helm.common.general import ensure_directory_exists, asdict_without_nones
from helm.common.hierarchical_logger import hlog


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


class SlurmRunner(Runner):
    """Runner that runs the entire benchmark on Slurm."""

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

    def run_all(self, run_specs: List[RunSpec]):
        """Run the entire benchmark on Slurm, where each RunSpec is run in its own Slurm job.

        This functions as a "manager" job that submits a Slurm worker job via sbatch for each RunSpec,
        and then monitors all the Slurm worker."""
        # This directory will be used for coordinating with workers
        ensure_directory_exists(self.slurm_base_dir)

        # Write the config to the directory
        slurm_runner_spec_json = json.dumps(asdict_without_nones(self.slurm_runner_spec), indent=2)
        slurm_runner_spec_path = os.path.join(self.slurm_base_dir, "slurm_runner_spec.json")
        hlog(f"Writing SlurmRunnerSpec to {slurm_runner_spec_path}")
        with open(slurm_runner_spec_path, "w") as f:
            f.write(slurm_runner_spec_json)

        # Write the run specs to the directory
        run_specs_dir = os.path.join(self.slurm_base_dir, "run_specs")
        ensure_directory_exists(run_specs_dir)
        logs_dir = os.path.join(self.slurm_base_dir, "logs")
        ensure_directory_exists(logs_dir)
        run_name_to_slurm_job_id: Dict[str, int] = {}
        run_name_to_slurm_job_state: Dict[str, str] = {}

        # Cleanup by cancelling all jobs during program termination or if an exception is raised.
        def cancel_all_jobs():
            """Cancels all submitted Slurm jobs that are in a non-terminal state."""
            for run_name, slurm_job_state in run_name_to_slurm_job_state.values():
                if slurm_job_state not in TERMINAL_SLURM_JOB_STATES:
                    cancel_slurm_job(run_name_to_slurm_job_id[run_name])

        try:
            # Submit a Slurm job for each RunSpecs.
            for run_spec in run_specs:
                # TODO: If skip_completed_runs is set and the run is completed, skip creating the Slurm worker job
                run_name = run_spec.name
                run_spec_json = json.dumps(asdict_without_nones(run_spec), indent=2)
                run_spec_path = os.path.join(run_specs_dir, f"{run_spec.name}.json")
                hlog(f"Writing RunSpec for run {run_name} to {run_spec_path}")
                with open(run_spec_path, "w") as f:
                    f.write(run_spec_json)
                log_path = os.path.join(logs_dir, f"{run_spec.name}.log")
                command = (
                    f"{sys.executable}"
                    f" -m {SlurmRunner.__module__}"
                    f" --slurm-runner-spec-path {slurm_runner_spec_path}"
                    f" --run-spec-path {run_spec_path}"
                )
                hlog(f"Submitting Slurm worker job for run {run_name} with command: {command}")
                slurm_job_id = submit_slurm_job(command=command, job_name=run_name, output_path=log_path)
                hlog(f"Slurm worker job submitted for run {run_name} with Slurm job ID: {slurm_job_id}")
                run_name_to_slurm_job_id[run_name] = slurm_job_id
                run_name_to_slurm_job_state[run_name] = SlurmJobState.PENDING

            # Monitor submitted Slurm jobs for RunSpecs until an exit condition is triggered.
            slurm_job_ids_path = os.path.join(self.slurm_base_dir, "slurm_job_ids.json")
            slurm_job_states_path = os.path.join(self.slurm_base_dir, "slurm_job_states.json")
            hlog(f"Slurm job IDs of all Slurm worker jobs: {json.dumps(run_name_to_slurm_job_id, indent=2)}")
            hlog(f"Writing Slurm worker job IDs to {slurm_job_ids_path}")
            with open(slurm_job_ids_path, "w") as f:
                f.write(json.dumps(run_name_to_slurm_job_id, indent=2))
            hlog("Entering Slurm worker job monitoring loop")
            while True:
                hlog("Getting states of Slurm worker jobs")
                for run_name, slurm_job_id in run_name_to_slurm_job_id.items():
                    run_name_to_slurm_job_state[run_name] = get_slurm_job_state(slurm_job_id)
                hlog(f"States of all Slurm worker jobs: {json.dumps(run_name_to_slurm_job_state, indent=2)}")
                hlog(f"Writing Slurm worker job states to {slurm_job_states_path}")
                with open(slurm_job_states_path, "w") as f:
                    f.write(json.dumps(run_name_to_slurm_job_state, indent=2))
                if self.exit_on_error and any(
                    [
                        slurm_job_state in FAILURE_SLURM_JOB_STATES
                        for slurm_job_state in run_name_to_slurm_job_state.values()
                    ]
                ):
                    hlog("Some Slurm worker job failed and --exit_on_error was set, exiting Slurm job monitoring loop.")
                    break
                if all(
                    [
                        slurm_job_state in TERMINAL_SLURM_JOB_STATES
                        for slurm_job_state in run_name_to_slurm_job_state.values()
                    ]
                ):
                    hlog("All Slurm worker jobs completed, exiting Slurm job monitoring loop.")
                    break
        finally:
            cancel_all_jobs()
        failed_run_names = [
            run_name
            for run_name, slurm_job_state in run_name_to_slurm_job_state.items()
            if slurm_job_state in FAILURE_SLURM_JOB_STATES
        ]
        if failed_run_names:
            failed_runs_str = ", ".join([f'"{run_name}"' for run_name in failed_run_names])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")


def run_as_worker(slurm_runner_spec_path: str, run_spec_path: str):
    """Deserialize SlurmRunner and RunSpec from the given files, then run the RunSpec with the SlurmRunner.

    Used by the worker Slurm jobs only."""
    with open(slurm_runner_spec_path, "r") as f:
        slurm_runner_spec = dacite.from_dict(SlurmRunnerSpec, json.load(f))
    with open(run_spec_path, "r") as f:
        run_spec = dacite.from_dict(RunSpec, json.load(f))
    slurm_runner = SlurmRunner(**slurm_runner_spec.to_kwargs())
    slurm_runner.run_one(run_spec)


def main():
    """Entry point for the worker Slurm jobs that run a single RunSpec."""
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
    run_as_worker(slurm_runner_spec_path=args.slurm_runner_spec_path, run_spec_path=args.run_spec_path)


if __name__ == "__main__":
    main()
