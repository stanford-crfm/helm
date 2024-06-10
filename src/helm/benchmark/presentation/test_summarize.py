import os
import tempfile

from helm.benchmark.presentation.summarize import Summarizer
from helm.benchmark.presentation.schema import get_default_schema_path
from helm.common.general import ensure_directory_exists


def test_summarize_suite():
    with tempfile.TemporaryDirectory() as output_path:
        ensure_directory_exists(os.path.join(output_path, "runs", "test_suite"))
        summarizer = Summarizer(
            release=None,
            suites=None,
            suite="test_suite",
            schema_path=get_default_schema_path(),
            output_path=output_path,
            verbose=False,
            num_threads=4,
            allow_unknown_models=True,
        )
        summarizer.run_pipeline(skip_completed=True, num_instances=1000)
        assert os.path.isfile(os.path.join(output_path, "runs", "test_suite", "groups.json"))


def test_summarize_release():
    with tempfile.TemporaryDirectory() as output_path:
        ensure_directory_exists(os.path.join(output_path, "runs", "test_suite_1"))
        ensure_directory_exists(os.path.join(output_path, "runs", "test_suite_2"))
        summarizer = Summarizer(
            release="test_release",
            suites=["test_suite_1", "test_suite_2"],
            suite=None,
            schema_path=get_default_schema_path(),
            output_path=output_path,
            verbose=False,
            num_threads=4,
            allow_unknown_models=True,
        )
        summarizer.run_pipeline(skip_completed=True, num_instances=1000)
        assert os.path.isfile(os.path.join(output_path, "releases", "test_release", "groups.json"))
