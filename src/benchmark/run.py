import argparse
import getpass
from typing import List, Dict, Any, Tuple

from common.hierarchical_logger import hlog, htrack, htrack_block
from common.authentication import Authentication
from .executor import ExecutionSpec
from .runner import Runner, RunSpec
from .test_utils import get_run_spec1, get_mmlu_spec


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
        default="http://crfm-models.stanford.edu",
        help="URL of the instance to use when benchmarking.",
    )
    parser.add_argument("-r", "--run-specs", nargs="*", help="Specifies what to run", default=["simple1"])
    parser.add_argument("-o", "--output-path", help="Where to save all the output", default="benchmark_output")
    args = parser.parse_args()

    api_key: str = getpass.getpass(prompt="Enter a valid API key: ")
    execution_spec = ExecutionSpec(auth=Authentication(api_key=api_key), url=args.url, parallelism=5)

    run_specs = [run_spec for description in args.run_specs for run_spec in parse_run_specs(description)]
    with htrack_block("Run specs"):
        for run_spec in run_specs:
            hlog(run_spec.scenario)

    runner = Runner(execution_spec, args.output_path, run_specs)
    runner.run_all()
