import argparse
from typing import List, Dict, Any, Tuple

from common.hierarchical_logger import hlog, htrack, htrack_block
from common.authentication import Authentication
from proxy.remote_service import create_authentication

from .executor import ExecutionSpec
from .runner import Runner, RunSpec
from .test_utils import (
    get_run_spec1,
    get_mmlu_spec,
    get_real_toxicity_prompts_spec,
    get_twitter_aae_spec,
)


def parse_run_specs(description: str) -> List[RunSpec]:
    """
    Parse `description` into a list of `RunSpec`s.
    `description` has the format:
        <name>:<key>=<value>,<key>=<value>
    """

    def parse_arg(arg: str) -> Tuple[str, Any]:
        key, value = arg.split("=", 1)
        return (key, value)

    if ":" in description:
        name, args_str = description.split(":", 1)
        args: Dict[str, Any] = dict(parse_arg(arg) for arg in args_str.split(","))
    else:
        name = description
        args = {}

    if name == "simple1":
        return [get_run_spec1()]
    if name == "mmlu":
        return [get_mmlu_spec(**args)]
    if name == "twitter_aae":
        return [get_twitter_aae_spec(**args)]
    if name == "real_toxicity_prompts":
        return [get_real_toxicity_prompts_spec()]
    raise ValueError(f"Unknown run spec: {description}")


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

    run_specs = [run_spec for description in args.run_specs for run_spec in parse_run_specs(description)]
    with htrack_block("Run specs"):
        for run_spec in run_specs:
            hlog(run_spec.scenario)

    runner = Runner(execution_spec, args.output_path, run_specs)
    runner.run_all()
