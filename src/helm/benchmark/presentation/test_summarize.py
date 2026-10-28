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


def _summarizer_with_overlapping_groups(tmp_path, extra_groups):
    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.metrics.metric_name import MetricName
    from helm.benchmark.metrics.statistic import Stat
    from helm.benchmark.presentation.schema import Field, MetricGroup, MetricNameMatcher, RunGroup, Schema
    from helm.benchmark.presentation.summarize import Run
    from helm.benchmark.run_spec import RunSpec
    from helm.benchmark.scenarios.scenario import ScenarioSpec

    ensure_directory_exists(os.path.join(tmp_path, "runs", "test_suite"))
    summarizer = Summarizer(None, None, "test_suite", None, str(tmp_path), False, 1, True)
    root = RunGroup(name="root", subgroups=["x", "y"], metric_groups=["accuracy"])
    summarizer.schema = Schema(
        metrics=[Field(name="exact_match", lower_is_better=False)],
        metric_groups=[
            MetricGroup(
                name="accuracy",
                metrics=[MetricNameMatcher("exact_match", "test")],
                aggregation_strategies=[],
            )
        ],
        run_groups=[
            root,
            RunGroup(name="x", metric_groups=["accuracy"]),
            RunGroup(name="y", metric_groups=["accuracy"]),
        ],
    )
    summarizer.read_runs()
    for name, groups, score, tokens in [("a", ["root"], 0, 10), ("b", ["root"] + extra_groups, 1, 20)]:
        summarizer.runs.append(
            Run(
                run_path=f"test_suite/{name}",
                run_spec=RunSpec(
                    name,
                    ScenarioSpec("test.scenario", {}),
                    AdapterSpec(model="simple/model1", model_deployment="simple/model1"),
                    [],
                    groups=groups,
                ),
                stats=[
                    Stat(MetricName("exact_match", split="test")).add(score),
                    Stat(MetricName("num_prompt_tokens")).add(tokens),
                ],
            )
        )
    summarizer.group_runs()
    return summarizer, root


def test_overlapping_group_labels_do_not_change_scores(tmp_path):
    for extra_groups in [[], ["x"], ["x", "y"]]:
        summarizer, root = _summarizer_with_overlapping_groups(tmp_path, extra_groups)
        table = summarizer.create_group_tables_by_metric_group(root)[0]
        root_column = next(
            i for i, header in enumerate(table.header) if header.metadata and header.metadata["run_group"] == "root"
        )
        cell = table.rows[0][root_column]
        assert cell.value == 0.5
        assert cell.description == "min=0, mean=0.5, max=1, sum=1 (2)"
        assert cell.run_spec_names == ["a", "b"]


def test_overlapping_group_labels_do_not_change_token_totals(tmp_path):
    for extra_groups in [[], ["x"], ["x", "y"]]:
        summarizer, _ = _summarizer_with_overlapping_groups(tmp_path, extra_groups)
        table = summarizer.create_index_tables()[0]
        tokens_column = next(i for i, header in enumerate(table.header) if header.value == "# prompt tokens")
        assert table.rows[0][tokens_column].value == 30


def test_summary_keeps_distinct_run_paths_with_the_same_name(tmp_path):
    from dataclasses import replace

    summarizer, root = _summarizer_with_overlapping_groups(tmp_path, ["x", "y"])
    # A release can contain the same run spec in multiple suites. Both observations count.
    original = summarizer.runs[0]
    summarizer.runs.append(replace(original, run_path="another_suite/a"))
    # Reset the group indexes without changing the runs under test.
    summarizer.group_adapter_to_runs.clear()
    summarizer.group_scenario_adapter_to_runs.clear()
    summarizer.group_runs()
    table = summarizer.create_group_tables_by_metric_group(root)[0]
    root_column = next(
        i for i, header in enumerate(table.header) if header.metadata and header.metadata["run_group"] == "root"
    )
    assert table.rows[0][root_column].value == 1 / 3
    index_table = summarizer.create_index_tables()[0]
    assert index_table.rows[0][5].value == 40
