import sys
import tempfile
import shutil
import textwrap
import pathlib
from helm.benchmark import run


def test_run_with_custom_logging_config():
    # Setup temporary directory
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    log_path = tmp_dir / "helm_test.log"
    log_config_path = tmp_dir / "mylog.yaml"

    sys_argv_backup = sys.argv[:]
    try:
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
        sys.argv = [
            "run.py",  # Fake script name
            "--run-entries", "mmlu:subject=philosophy,model=openai/gpt2",
            "-m", "1",
            "--suite", "my-suite",
            "--dry-run",
            "--log-config",
            str(log_config_path),
        ]

        # Call main
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

    finally:
        # Restore sys.argv
        sys.argv = sys_argv_backup
        # Clean up temp files
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_run_with_custom_logging_config()
