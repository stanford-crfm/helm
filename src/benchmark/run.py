import argparse
from typing import List

from common.hierarchical_logger import hlog, htrack, htrack_block
from common.authentication import Authentication
from common.object_spec import parse_object_spec
from proxy.remote_service import create_authentication, add_service_args

from .executor import ExecutionSpec
from .runner import Runner
from .run_specs import construct_run_specs


def run_benchmarking(
    run_spec_descriptions: List[str], auth: Authentication, url: str, num_threads: int, output_path: str, dry_run: bool
):
    """Runs RunSpecs given a list of RunSpec descriptions."""
    execution_spec = ExecutionSpec(auth=auth, url=url, parallelism=num_threads, dry_run=dry_run)

    run_specs = [
        run_spec
        for description in run_spec_descriptions
        for run_spec in construct_run_specs(parse_object_spec(description))
    ]
    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec.scenario)

    runner = Runner(execution_spec, output_path, run_specs)
    runner.run_all()


@htrack(None)
def main():
    """
    Main entry point for running the benchmark.
    """
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument("-r", "--run-specs", nargs="*", help="Specifies what to run", default=["simple1"])
    parser.add_argument("-o", "--output-path", help="Where to save all the output", default="benchmark_output")
    parser.add_argument("-n", "--num-threads", type=int, help="Max number of threads to make requests", default=5)
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="If set, the runner will skip execution, dump the scenario states and estimate token usage.",
    )
    args = parser.parse_args()

    auth: Authentication = create_authentication(args)
    run_benchmarking(
        args.run_specs,
        auth=auth,
        url=args.url,
        num_threads=args.num_threads,
        output_path=args.output_path,
        dry_run=args.dry_run,
    )
