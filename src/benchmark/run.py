import argparse
from typing import List, Dict, Any, Tuple

from common.hierarchical_logger import hlog, htrack, htrack_block
from common.authentication import Authentication
from common.object_spec import parse_object_spec
from proxy.remote_service import create_authentication

from .executor import ExecutionSpec
from .runner import Runner, RunSpec
from .run_specs import construct_run_specs

@htrack(None)
def main():
    """
    Main entry point for running the benchmark.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="https://crfm-models.stanford.edu",
        help="URL of the instance to use when benchmarking",
    )
    parser.add_argument(
        "-a", "--api-key-path", type=str, default="proxy_api_key.txt", help="Path to API key",
    )
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
    execution_spec = ExecutionSpec(auth=auth, url=args.url, parallelism=args.num_threads, dry_run=args.dry_run)

    run_specs = [run_spec for description in args.run_specs for run_spec in construct_run_specs(parse_object_spec(description))]
    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec.scenario)

    runner = Runner(execution_spec, args.output_path, run_specs)
    runner.run_all()
