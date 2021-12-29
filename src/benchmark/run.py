from common.authentication import Authentication
from .executor import ExecutionSpec
from .runner import Runner
from .test_utils import get_run_spec1


def main():
    """
    Main entry point for running the benchmark.
    """
    # TODO: make a general client for this
    execution_spec = ExecutionSpec(
        auth=Authentication(username="crfm", password="crfm"),
        url="http://localhost:1959/api/makeRequest",
        parallelism=5,
    )

    # TODO: load this dynamically from a conf file, for now, just use a test.
    run_spec = get_run_spec1()

    runner = Runner(execution_spec, [run_spec])
    runner.run_all()


main()
