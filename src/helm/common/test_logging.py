import sys
import tempfile
import textwrap
import pathlib
from helm.benchmark import run
from typing import List, Optional


class ArgvContext:
    """
    Helper to assign a temporary value to sys.argv and then restore it
    """

    def __init__(self, argv: Optional[List[str]]):
        self.argv = argv
        self._original_argv: Optional[List[str]] = None

    def __enter__(self):
        self._original_argv = sys.argv[:]
        sys.argv = self.argv or []

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._original_argv is not None  # Satisfies mypy
        sys.argv = self._original_argv


def test_run_with_custom_logging_config():
    # Setup temporary directory
    with tempfile.TemporaryDirectory(prefix="helm_test_") as tmp_dir_str:
        tmp_dir = pathlib.Path(tmp_dir_str)
        log_path = tmp_dir / "test.log"
        log_config_path = tmp_dir / "test_config.yaml"

        # Write custom YAML log config to file
        log_config_text = textwrap.dedent(
            f"""
            version: 1
            disable_existing_loggers: false
            formatters:
              simple:
                datefmt: '%Y-%m-%dT%H:%M:%S'
                format: '%(asctime)s %(levelname)s %(name)s %(message)s'
            handlers:
              file:
                class: logging.FileHandler
                filename: {log_path}
                formatter: simple
                level: DEBUG
                mode: w
            loggers:
              helm:
                handlers:
                - file
                level: DEBUG
                propagate: false
            """
        ).strip()

        log_config_path.write_text(log_config_text)

        # Simulate command-line arguments
        argv = [
            "run.py",  # Fake script name
            "--run-entries",
            "mmlu:subject=philosophy,model=openai/gpt2",
            "-m",
            "1",
            "--suite",
            "my-suite",
            "--dry-run",
            "--log-config",
            str(log_config_path),
        ]

        # Call main
        with ArgvContext(argv):
            run.main()

        # Check log file contents
        assert log_path.exists(), "Log file was not created"
        log_contents = log_path.read_text()

        # Test that log file was written to disk as requested
        print("Log Contents")
        print("------------")
        print(log_contents)

        assert (
            "mscoco" in log_contents or "huggingface" in log_contents or "dry-run" in log_contents
        ), "Expected log content not found in log file:\n"


if __name__ == "__main__":
    test_run_with_custom_logging_config()
