import argparse
from dataclasses import replace
from typing import List, Optional

from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.common.authentication import Authentication
from helm.common.object_spec import parse_object_spec
from helm.proxy.services.remote_service import create_authentication, add_service_args

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .executor import ExecutionSpec
from .runner import Runner, RunSpec
from .run_specs import construct_run_specs


LATEST_SYMLINK: str = "latest"


def run_benchmarking(
    run_spec_descriptions: List[str],
    auth: Authentication,
    url: str,
    local: bool,
    local_path: str,
    num_threads: int,
    output_path: str,
    suite: str,
    dry_run: bool,
    skip_instances: bool,
    max_eval_instances: Optional[int] = None,
    num_train_trials: Optional[int] = None,
    groups: Optional[List[str]] = None,
    models_to_run: Optional[List[str]] = None,
    groups_to_run: Optional[List[str]] = None,
    mongo_uri: str = "",
) -> List[RunSpec]:
    """Runs RunSpecs given a list of RunSpec descriptions."""

    execution_spec = ExecutionSpec(
        auth=auth,
        url=url,
        local=local,
        local_path=local_path,
        parallelism=num_threads,
        dry_run=dry_run,
        mongo_uri=mongo_uri,
    )

    def override(run_spec: RunSpec) -> RunSpec:
        """Override parts of `run_spec`."""
        # Modify AdapterSpec
        adapter_spec: AdapterSpec = run_spec.adapter_spec
        if max_eval_instances is not None:
            adapter_spec = replace(adapter_spec, max_eval_instances=max_eval_instances)
        if num_train_trials is not None or adapter_spec.max_train_instances == 0:
            adapter_spec = replace(
                adapter_spec, num_train_trials=1 if adapter_spec.max_train_instances == 0 else num_train_trials
            )

        run_spec = replace(run_spec, adapter_spec=adapter_spec)

        # Append groups
        if groups is not None:
            groups_name: str = "" if len(groups) == 0 else f",groups={'-'.join(sorted(groups))}"
            run_spec = replace(run_spec, name=run_spec.name + groups_name, groups=run_spec.groups + groups)

        return run_spec

    run_specs = [
        override(run_spec)
        for description in run_spec_descriptions
        for run_spec in construct_run_specs(parse_object_spec(description))
        if (not models_to_run or run_spec.adapter_spec.model in models_to_run)
        and (not groups_to_run or any(group in groups_to_run for group in run_spec.groups))
    ]

    if len(run_specs) == 0:
        return run_specs

    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec.name)

    runner = Runner(execution_spec, output_path, suite, run_specs, skip_instances)
    runner.run_all()
    return run_specs


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where to save all the output", default="benchmark_output"
    )
    parser.add_argument("-n", "--num-threads", type=int, help="Max number of threads to make requests", default=4)
    parser.add_argument(
        "--skip-instances",
        action="store_true",
        default=None,
        help="Skip creation of instances (basically do nothing but just parse everything).",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skip execution, only output scenario states and estimate token usage.",
    )
    parser.add_argument(
        "-m",
        "--max-eval-instances",
        type=int,
        required=True,
        help="Maximum number of instances to evaluate on, overrides the value in Adapter spec.",
    )
    parser.add_argument(
        "-t",
        "--num-train-trials",
        type=int,
        help="Number of trials where each trial samples a different set of in-context examples. "
        "Overrides the value in Adapter spec.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this run belongs to (default is today's date).",
        required=True,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If true, bypasses the proxy server and runs everything locally",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="If running locally, the path for `ServerService`.",
        default="prod_env",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="If non-empty, the URL of the MongoDB database that will be used for caching instead of SQLite",
        default="",
    )


def validate_args(args):
    assert args.suite != LATEST_SYMLINK, f"Suite name can't be '{LATEST_SYMLINK}'"


@htrack(None)
def main():
    """
    Main entry point for running the benchmark.
    """
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument("-r", "--run-specs", nargs="*", help="Specifies what to run", default=["simple1"])
    add_run_args(parser)
    args = parser.parse_args()
    validate_args(args)

    auth: Authentication = Authentication("") if args.skip_instances or args.local else create_authentication(args)
    run_benchmarking(
        args.run_specs,
        auth=auth,
        url=args.server_url,
        local=args.local,
        local_path=args.local_path,
        num_threads=args.num_threads,
        output_path=args.output_path,
        suite=args.suite,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        max_eval_instances=args.max_eval_instances,
        mongo_uri=args.mongo_uri,
    )


if __name__ == "__main__":
    main()
