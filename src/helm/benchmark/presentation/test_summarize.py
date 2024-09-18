import os
import tempfile

from helm.benchmark.presentation.summarize import Summarizer, compute_aggregate_row_win_rates
from helm.benchmark.presentation.schema import get_default_schema_path
from helm.benchmark.presentation.table import Cell, HeaderCell, Table
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


def test_compute_win_rates_one_scenario():
    values = [
        ["Model A", 1],
        ["Model B", 2],
        ["Model C", 3],
        ["Model D", 4],
        ["Model E", 5],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_two_scenarios():
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, 2],
        ["Model D", 4, 5],
        ["Model E", 5, 4],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.25, 0.125, 0.375, 0.875, 0.875]


def test_compute_win_rates_incomplete_values():
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, None],
        ["Model D", 4, None],
        ["Model E", 5, None],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.5, 0.125, 0.5, 0.75, 1]


def test_compute_win_rates_ignore_nones():
    values = [
        ["Model A", 1, None, None],
        ["Model B", 2, None, 1],
        ["Model C", 3, None, None],
        ["Model D", 4, None, None],
        ["Model E", 5, None, None],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
        HeaderCell(value="Scenario C", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_ignore_unset_lower_is_better():
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, 2],
        ["Model D", 4, 5],
        ["Model E", 5, 4],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B"),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_no_win_rate():
    values = [
        ["Model A", None],
        ["Model B", None],
        ["Model C", None],
        ["Model D", None],
        ["Model E", None],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [None, None, None, None, None]


def test_compute_win_rates_ties():
    values = [
        ["Model A", 1],
        ["Model B", 1],
        ["Model C", 1],
        ["Model D", 4],
        ["Model E", 5],
    ]
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.0, 0.25, 0.5, 0.75, 1.0]
