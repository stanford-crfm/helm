import argparse
import os
import traceback

from tqdm import tqdm
from typing import List, Optional

from helm.common.authentication import Authentication
from helm.common.general import write_lines
from helm.common.hierarchical_logger import hlog, htrack
from helm.benchmark.run import run_benchmarking, add_run_args, validate_args, LATEST_SYMLINK
from helm.benchmark.runner import RunSpec
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.proxy.services.remote_service import add_service_args, create_authentication

"""
Runs all the RunSpecs in run_specs.conf and outputs JSON files.
TODO: rename this file to `run_all.py`

Usage:

    venv/bin/helm-run

"""


class AllRunner:
    """Runs all RunSpecs specified in the configuration file."""

    def __init__(
        self,
        auth: Authentication,
        conf_paths: List[str],
        url: str,
        local: bool,
        local_path: str,
        output_path: str,
        suite: str,
        num_threads: int,
        dry_run: Optional[bool],
        skip_instances: bool,
        max_eval_instances: Optional[int],
        num_train_trials: Optional[int],
        models_to_run: Optional[List[str]],
        groups_to_run: Optional[List[str]],
        exit_on_error: bool,
        priority: Optional[int],
        mongo_uri: str,
    ):
        self.auth: Authentication = auth
        self.conf_paths: List[str] = conf_paths
        self.url: str = url
        self.local: bool = local
        self.local_path: str = local_path
        self.output_path: str = output_path
        self.suite: str = suite
        self.num_threads: int = num_threads
        self.dry_run: Optional[bool] = dry_run
        self.skip_instances: bool = skip_instances
        self.max_eval_instances: Optional[int] = max_eval_instances
        self.num_train_trials: Optional[int] = num_train_trials
        self.models_to_run: Optional[List[str]] = models_to_run
        self.groups_to_run: Optional[List[str]] = groups_to_run
        self.exit_on_error: bool = exit_on_error
        self.priority: Optional[int] = priority
        self.mongo_uri = mongo_uri

    @htrack(None)
    def run(self):
        run_specs: List[RunSpec] = []
        runs_dir: str = os.path.join(self.output_path, "runs")
        suite_dir: str = os.path.join(runs_dir, self.suite)

        run_entries = read_run_entries(self.conf_paths)

        for entry in tqdm(run_entries.entries):
            # Filter by priority
            priority: int = entry.priority
            if self.priority is not None and priority > self.priority:
                continue

            try:
                new_run_specs = run_benchmarking(
                    run_spec_descriptions=[entry.description],
                    auth=self.auth,
                    url=self.url,
                    local=self.local,
                    local_path=self.local_path,
                    num_threads=self.num_threads,
                    output_path=self.output_path,
                    suite=self.suite,
                    dry_run=self.dry_run,
                    skip_instances=self.skip_instances,
                    max_eval_instances=self.max_eval_instances,
                    num_train_trials=self.num_train_trials,
                    groups=entry.groups,
                    models_to_run=self.models_to_run,
                    groups_to_run=self.groups_to_run,
                    mongo_uri=self.mongo_uri,
                )
                run_specs.extend(new_run_specs)

            except Exception as e:
                if self.exit_on_error:
                    raise e
                else:
                    hlog(f"Error when running {entry.description}:\n{traceback.format_exc()}")

        if len(run_specs) == 0:
            hlog("There were no RunSpecs or they got filtered out.")
            return

        hlog(f"{len(run_entries.entries)} entries produced into {len(run_specs)} run specs")

        if self.skip_instances:
            self.write_parallel_commands(suite_dir, run_specs)

        # Create a symlink runs/latest -> runs/<name_of_suite>,
        # so runs/latest always points to the latest run suite.
        symlink_path: str = os.path.abspath(os.path.join(runs_dir, LATEST_SYMLINK))
        if os.path.islink(symlink_path):
            # Remove the previous symlink if it exists.
            os.unlink(symlink_path)
        os.symlink(os.path.abspath(suite_dir), symlink_path)

    def write_parallel_commands(self, suite_dir: str, run_specs: List[RunSpec]):
        """
        Print out scripts to run after.
        """
        # Print out all the models and groups that we're touching.
        models = set()
        groups = set()
        for run_spec in run_specs:
            models.add(run_spec.adapter_spec.model)
            for group in run_spec.groups:
                groups.add(group)
        hlog(f"{len(models)} models: {' '.join(models)}")
        hlog(f"{len(groups)} groups: {' '.join(groups)}")

        # Write wrapper for helm-run that can be used through Slurm
        lines = [
            "#!/bin/bash",
            "",
            ". venv/bin/activate",
            'helm-run "$@"',
        ]
        write_lines(os.path.join(suite_dir, "helm-run.sh"), lines)

        # Write out bash script for launching the entire benchmark
        lines = []
        for model in models:
            for group in groups:
                # Try to match the arguments of `run_benchmarking`
                # Build arguments
                present_args = []
                present_args.append(f"--confs {' '.join(self.conf_paths)}")
                if self.local:
                    present_args.append("--local")
                present_args.append(f"--num-threads {self.num_threads}")
                present_args.append(f"--suite {self.suite}")
                if self.max_eval_instances is not None:
                    present_args.append(f"--max-eval-instances {self.max_eval_instances}")
                present_args.append(f"--models-to-run {model}")
                present_args.append(f"--scenario-groups-to-run {group}")

                lines.append(
                    f"sbatch --partition john "
                    f"--cpus {self.num_threads} "
                    f"-o benchmark_output/runs/{self.suite}/slurm-%j.out "
                    f"{suite_dir}/helm-run.sh "
                    f"{' '.join(present_args)}"
                )
        lines.append("echo '# Run these after Slurm jobs terminate'")
        lines.append(f"echo 'helm-run --local --suite {self.suite} --skip-instances'")
        lines.append(f"echo 'helm-summarize --suite {self.suite}'")
        write_lines(os.path.join(suite_dir, "run-all.sh"), lines)


def main():
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "-c",
        "--conf-paths",
        nargs="+",
        help="Where to read RunSpecs to run from",
        default=["src/helm/benchmark/presentation/run_specs.conf"],
    )
    parser.add_argument(
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, runs with all models.",
        default=None,
    )
    parser.add_argument(
        "--groups-to-run",
        nargs="+",
        help="Only RunSpecs with these (scenario) groups specified. " "If no group is specified, runs with all groups.",
        default=None,
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        default=None,
        help="Fail and exit immediately if a particular RunSpec fails.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Run RunSpecs with priority less than or equal to this number. "
        "If a value for --priority is not specified, run on everything",
    )
    add_run_args(parser)
    args = parser.parse_args()
    validate_args(args)

    runner = AllRunner(
        # Use a dummy API key when `skip_instances` or `local` is set.
        # The benchmarking framework will not make any requests to the proxy server when
        # `skip_instances` is set, so a valid API key is not necessary.
        # Setting `local` will run and cache everything locally.
        auth=Authentication("") if args.skip_instances or args.local else create_authentication(args),
        conf_paths=args.conf_paths,
        url=args.server_url,
        local=args.local,
        local_path=args.local_path,
        output_path=args.output_path,
        suite=args.suite,
        num_threads=args.num_threads,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        max_eval_instances=args.max_eval_instances,
        num_train_trials=args.num_train_trials,
        models_to_run=args.models_to_run,
        groups_to_run=args.groups_to_run,
        exit_on_error=args.exit_on_error,
        priority=args.priority,
        mongo_uri=args.mongo_uri,
    )

    # Run the benchmark!
    runner.run()

    hlog("Done.")


if __name__ == "__main__":
    main()
