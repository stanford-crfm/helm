import argparse
from typing import List, Optional
from dataclasses import replace

from common.hierarchical_logger import hlog, htrack, htrack_block
from common.authentication import Authentication
from common.object_spec import parse_object_spec
from proxy.remote_service import create_authentication, add_service_args

from .executor import ExecutionSpec
from .runner import Runner, RunSpec
from .run_specs import construct_run_specs


def run_benchmarking(
    run_spec_descriptions: List[str],
    auth: Authentication,
    url: str,
    num_threads: int,
    output_path: str,
    dry_run: bool,
    max_eval_instances: Optional[int],
):
    """Runs RunSpecs given a list of RunSpec descriptions."""
    execution_spec = ExecutionSpec(auth=auth, url=url, parallelism=num_threads, dry_run=dry_run)

    def override(run_spec: RunSpec) -> RunSpec:
        """Override parts of `run_spec`."""
        if max_eval_instances is not None:
            run_spec = replace(
                run_spec, adapter_spec=replace(run_spec.adapter_spec, max_eval_instances=max_eval_instances)
            )
        return run_spec

    run_specs = [
        override(run_spec)
        for description in run_spec_descriptions
        for run_spec in construct_run_specs(parse_object_spec(description))
    ]
    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec.scenario)

    runner = Runner(execution_spec, output_path, run_specs)
    runner.run_all()


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output-path", help="Where to save all the output", default="benchmark_output")
    parser.add_argument("-n", "--num-threads", type=int, help="Max number of threads to make requests", default=5)
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
        help="Maximum number of instances to evaluate on, overrides adapter spec (for piloting)",
    )


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

    auth: Authentication = create_authentication(args)
    run_benchmarking(
        args.run_specs,
        auth=auth,
        url=args.server_url,
        num_threads=args.num_threads,
        output_path=args.output_path,
        dry_run=args.dry_run,
        max_eval_instances=args.max_eval_instances,
    )
