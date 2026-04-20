import os
import tempfile

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.presentation.schema import Field, MetricGroup, MetricNameMatcher, RunGroup, Schema
from helm.benchmark.presentation.summarize import Summarizer, compute_aggregate_row_win_rates
from helm.benchmark.presentation.summarize import Run
from helm.benchmark.presentation.schema import get_default_schema_path
from helm.benchmark.presentation.table import Cell, HeaderCell, Table
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.scenarios.scenario import ScenarioSpec
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
        summarizer.run_pipeline(skip_completed=True)
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
        summarizer.run_pipeline(skip_completed=True)
        assert os.path.isfile(os.path.join(output_path, "releases", "test_release", "groups.json"))


def test_compute_win_rates_one_scenario():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    values = [
        ["Model A", 1],
        ["Model B", 2],
        ["Model C", 3],
        ["Model D", 4],
        ["Model E", 5],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_two_scenarios():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
    ]
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, 2],
        ["Model D", 4, 5],
        ["Model E", 5, 4],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.25, 0.125, 0.375, 0.875, 0.875]


def test_compute_win_rates_incomplete_values():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
    ]
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, None],
        ["Model D", 4, None],
        ["Model E", 5, None],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.5, 0.125, 0.5, 0.75, 1]


def test_compute_win_rates_ignore_nones():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B", lower_is_better=False),
        HeaderCell(value="Scenario C", lower_is_better=False),
    ]
    values = [
        ["Model A", 1, None, None],
        ["Model B", 2, None, 1],
        ["Model C", 3, None, None],
        ["Model D", 4, None, None],
        ["Model E", 5, None, None],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_ignore_unset_lower_is_better():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
        HeaderCell(value="Scenario B"),
    ]
    values = [
        ["Model A", 1, 3],
        ["Model B", 2, 1],
        ["Model C", 3, 2],
        ["Model D", 4, 5],
        ["Model E", 5, 4],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0, 0.25, 0.5, 0.75, 1]


def test_compute_win_rates_no_win_rate():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    values = [
        ["Model A", None],
        ["Model B", None],
        ["Model C", None],
        ["Model D", None],
        ["Model E", None],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [None, None, None, None, None]


def test_compute_win_rates_ties():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=False),
    ]
    values = [
        ["Model A", 1],
        ["Model B", 1],
        ["Model C", 1],
        ["Model D", 4],
        ["Model E", 5],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [0.25, 0.25, 0.25, 0.75, 1.0]


def test_compute_win_rates_lower_is_better():
    header = [
        HeaderCell(value="Model"),
        HeaderCell(value="Scenario A", lower_is_better=True),
    ]
    values = [
        ["Model A", 1],
        ["Model B", 2],
        ["Model C", 3],
        ["Model D", 4],
        ["Model E", 5],
    ]
    rows = [[Cell(value) for value in row_values] for row_values in values]
    table = Table(title="Test Table", header=header, rows=rows)
    assert compute_aggregate_row_win_rates(table) == [1, 0.75, 0.5, 0.25, 0]


def test_create_group_table_deduplicates_main_metric_columns():
    with tempfile.TemporaryDirectory() as output_path:
        summarizer = Summarizer(
            release=None,
            suites=None,
            suite="test_suite",
            schema_path=get_default_schema_path(),
            output_path=output_path,
            verbose=False,
            num_threads=1,
            allow_unknown_models=True,
        )
        summarizer.schema = Schema(
            metrics=[Field(name="accuracy", display_name="Accuracy", short_display_name="Accuracy")],
            metric_groups=[
                MetricGroup(
                    name="main_metric",
                    display_name="Main Metric",
                    metrics=[MetricNameMatcher(name="${main_name}", split="${main_split}")],
                ),
                MetricGroup(
                    name="accuracy_group",
                    display_name="Accuracy Group",
                    metrics=[MetricNameMatcher(name="accuracy", split="test")],
                ),
            ],
            run_groups=[
                RunGroup(
                    name="scenario_a",
                    display_name="Scenario A",
                    metric_groups=["main_metric", "accuracy_group"],
                    environment={"main_name": "accuracy", "main_split": "test"},
                )
            ],
        )

        adapter_spec = AdapterSpec(model="openai/gpt-4")
        run = Run(
            run_path="runs/test_suite/run_1",
            run_spec=RunSpec(
                name="run_1",
                scenario_spec=ScenarioSpec(class_name="test.scenario"),
                adapter_spec=adapter_spec,
                metric_specs=[],
                groups=["scenario_a"],
            ),
            stats=[Stat(MetricName(name="accuracy", split="test")).add(0.75)],
        )
        adapter_to_runs = {adapter_spec: [run]}

        table = summarizer.create_group_table(
            name="scenario_a",
            title="Scenario A",
            adapter_to_runs=adapter_to_runs,
            is_scenario_table=True,
            columns=[
                (summarizer.schema.run_groups[0], "main_metric"),
                (summarizer.schema.run_groups[0], "accuracy_group"),
            ],
            aggregation_strategies=[],
        )

        assert [cell.value for cell in table.header] == ["Model", "Accuracy"]
