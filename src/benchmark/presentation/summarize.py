import argparse
from dataclasses import dataclass, replace
import os
from collections import defaultdict
import urllib.parse

import dacite
from typing import List, Optional, Dict
import json

from common.general import write, ensure_directory_exists, asdict_without_nones, singleton
from common.hierarchical_logger import hlog, htrack
from benchmark.metrics.statistic import Stat
from benchmark.runner import RunSpec
from proxy.models import ALL_MODELS, get_model
from .table import Cell, Table, Hyperlink, table_to_latex
from .schema import MetricNameMatcher, ScenarioGroup, read_schema, SCHEMA_YAML_PATH
from .contamination import read_contamination, validate_contamination, CONTAMINATION_SYMBOLS, CONTAMINATION_STYLES

"""
Reads the output of the benchmark runs and produces:
- JSON files for the frontend
- Tables for the paper

Usage:

    venv/bin/benchmark-summarize --suite <Name of the suite>

"""


@dataclass(frozen=True)
class Run:
    """ Represents a run with spec and stats. """

    # Directory name of the run (used by frontend to find the actual instances to load)
    run_path: str

    # Run spec for the run
    run_spec: RunSpec

    # Statistics for the run
    stats: List[Stat]


def get_unique_stat_by_matcher(stats: List[Stat], matcher: MetricNameMatcher) -> Optional[Stat]:
    """Return the single stat that matches."""
    matching_stats = [stat for stat in stats if matcher.matches(stat.name)]
    if len(matching_stats) == 0:
        return None
    return singleton(matching_stats)


def get_benchmarking_url(params: Dict[str, str]) -> str:
    # Don't encode ' ' as '+'
    return "benchmarking.html?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


class Summarizer:
    """Summarize the benchmark results in JSON files to be displayed in the UI."""

    def __init__(self, run_suite_path: str):
        self.run_suite_path: str = run_suite_path

        self.schema = read_schema()
        self.contamination = read_contamination()
        validate_contamination(self.contamination, self.schema)

    def read_run(self, run_path: str) -> Run:
        """Load the `Run` object from `run_path`."""

        with open(os.path.join(run_path, "run_spec.json")) as f:
            run_spec = dacite.from_dict(RunSpec, json.load(f))

        with open(os.path.join(run_path, "stats.json")) as f:
            stats = [dacite.from_dict(Stat, raw) for raw in json.load(f)]

        return Run(run_path=run_path, run_spec=run_spec, stats=stats,)

    @htrack(None)
    def read_runs(self):
        """ Load the corresponding runs for the run specs in run_specs.json. """

        run_specs_path: str = os.path.join(self.run_suite_path, "run_specs.json")
        if not os.path.exists(run_specs_path):
            hlog(f"Summarizer won't run because {run_specs_path} doesn't exist yet. This is expected in a dry run.")
            return []

        self.runs: List[Run] = []
        with open(run_specs_path) as f:
            raw_run_specs = json.load(f)
        for raw_run_spec in raw_run_specs:
            run_spec = dacite.from_dict(RunSpec, raw_run_spec)
            run_path: str = os.path.join(self.run_suite_path, run_spec.name)

            run_spec_path: str = os.path.join(run_path, "run_spec.json")
            stats_path: str = os.path.join(run_path, "stats.json")

            if os.path.exists(run_spec_path) and os.path.exists(stats_path):
                run = self.read_run(run_path)
                self.runs.append(run)
            else:
                hlog(f"WARNING: {run_path} doesn't have run_spec.json or stats.json, skipping")

        # Build mapping from scenario group (e.g., natural_qa) and model (e.g., openai/davinci) to list of runs
        self.group_model_to_runs: Dict[str, Dict[str, List[Run]]] = defaultdict(lambda: defaultdict(list))
        self.group_scenario_model_to_runs: Dict[str, Dict[str, List[Run]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for run in self.runs:
            for group in run.run_spec.groups:
                model = run.run_spec.adapter_spec.model
                # Assume it is sensible to shard by scenario arguments.
                scenario = ", ".join(f"{k}: {v}" for k, v in run.run_spec.scenario_spec.args.items())

                self.group_model_to_runs[group][model].append(run)
                self.group_scenario_model_to_runs[group][scenario][model].append(run)

    @htrack(None)
    def check_metrics_defined(self):
        """Check that all the metrics that appear in stats are defined."""
        # Compute all metric names that were encountered
        metric_name_to_run_spec_names: Dict[str, List[str]] = defaultdict(list)
        for run in self.runs:
            for stat in run.stats:
                metric_name_to_run_spec_names[stat.name.name].append(run.run_spec.name)

        defined_metric_names = set(entry.name for entry in self.schema.metrics)

        for metric_name, run_spec_names in metric_name_to_run_spec_names.items():
            if metric_name not in defined_metric_names:
                hlog(
                    f"WARNING: metric name {metric_name} undefined in {SCHEMA_YAML_PATH} "
                    f"but appears in {len(run_spec_names)} run specs, including {run_spec_names[0]}"
                )

    def write_models(self):
        write(
            os.path.join(self.run_suite_path, "models.json"),
            json.dumps(list(map(asdict_without_nones, ALL_MODELS)), indent=2),
        )

    def write_runs(self):
        write(
            os.path.join(self.run_suite_path, "runs.json"),
            json.dumps(list(map(asdict_without_nones, self.runs)), indent=2),
        )

    def create_index_table(self) -> Table:
        header = [
            Cell("Scenario"),
            Cell("Description"),
            Cell("# models"),
        ]
        rows = []
        for group in self.schema.scenario_groups:
            num_runs = len(self.group_model_to_runs[group.name])
            if num_runs == 0:
                continue
            rows.append(
                [
                    Cell(group.display_name, href=get_benchmarking_url({"group": group.name})),
                    Cell(group.description),
                    Cell(num_runs),
                ]
            )
        return Table(title="Overview of results", header=header, rows=rows)

    def create_cell(self, runs: List[Run], matcher: MetricNameMatcher, contamination_level: Optional[str]) -> Cell:
        """
        Use the metric name identified by `matcher` to pull out the stats from
        `runs` and return a representation of the average.
        """
        if len(runs) == 0:
            return Cell(None)

        aggregate_stat: Optional[Stat] = None
        for run in runs:
            stat = get_unique_stat_by_matcher(run.stats, matcher)
            if stat is None:
                hlog(f"WARNING: {matcher} doesn't match {run.run_spec.name}")
                continue  # TODO: probably should make a note that stats are missing
            stat = stat.take_mean()  # Collapse to a single point

            if aggregate_stat is None:
                aggregate_stat = replace(stat)  # Important: copy!
            else:
                aggregate_stat.merge(stat)

        if aggregate_stat is None:
            return Cell(None)

        value = aggregate_stat.mean
        display_value = round(value, 3) if value else value
        description = aggregate_stat.bare_str()
        style = CONTAMINATION_STYLES.get(contamination_level, {})
        return Cell(value=value, display_value=display_value, description=description, style=style)

    def create_group_table(
        self, title: str, scenario_group: ScenarioGroup, model_to_runs: Dict[str, List[Run]], link_to_runs: bool
    ) -> Table:
        """
        Create a table for a scenario_group (natural_qa) where each row is a
        model and columns are constructed based on metrics.
        """

        # Figure out what the columns of the table are.
        # Create header (cells to display) and the list of metric name filters
        # (to pull out information later).
        header: List[Cell] = []
        matchers: List[MetricNameMatcher] = []

        header.append(Cell("Model"))
        for metric_group_name in scenario_group.metric_groups:
            metric_group = self.schema.name_to_metric_group[metric_group_name]
            for metric in metric_group.metrics:
                matcher = metric.substitute(scenario_group.environment)
                header_field = self.schema.name_to_metric.get(matcher.name)
                if header_field is None:
                    hlog(f"WARNING: unknown metric name {matcher.name}, skipping")
                    continue

                header_name = header_field.get_short_display_name()
                description = header_field.display_name + ": " + header_field.description

                if matcher.perturbation_name is not None:
                    perturbation_field = self.schema.name_to_perturbation[matcher.perturbation_name]
                    header_name += " (" + perturbation_field.get_short_display_name() + ")"
                    description += (
                        "\n- Perturbation "
                        + perturbation_field.display_name
                        + ": "
                        + (perturbation_field.description or "???")
                    )

                header.append(Cell(header_name, description=description))
                matchers.append(matcher)

        def run_spec_names_to_url(run_spec_names: List[str]) -> str:
            return get_benchmarking_url(
                {
                    "runSpecs": json.dumps(run_spec_names),
                    "scenarioDisplayName": title,
                    "scenarioDescription": scenario_group.description,
                }
            )

        # Populate the contents of the table
        rows = []
        for model_name, runs in model_to_runs.items():
            model = get_model(model_name)

            # Link to all the runs under this model
            if link_to_runs:
                run_spec_names = [run.run_spec.name for run in runs]
                href = run_spec_names_to_url(run_spec_names)
            else:
                href = None

            # Render contamination information
            point = self.contamination.get_point(model_name, scenario_group.name)
            if point is not None:
                suffix = CONTAMINATION_SYMBOLS[point.level]  # Append to name of model
                description = point.description
                contamination_level = point.level
            else:
                suffix = ""
                description = ""
                contamination_level = None

            rows.append(
                [Cell(model.display_name + suffix, description=description, href=href)]
                + [self.create_cell(runs, matcher, contamination_level) for matcher in matchers]
            )

        # Link to all runs under all models (to compare models)
        all_run_spec_names = []
        for runs in model_to_runs.values():
            for run in runs:
                all_run_spec_names.append(run.run_spec.name)
        links = [Hyperlink(text="all models", href=run_spec_names_to_url(all_run_spec_names))]

        return Table(title=title, header=header, rows=rows, links=links)

    def write_groups(self):
        """
        Each group selects out a set of runs.

        For each group, output:
        - Main table (model x columns): each row aggregate over all runs that match the (group, model).
        - Table for each scenario spec.
        """

        # Write out index file with all the groups and basic stats
        write(
            os.path.join(self.run_suite_path, "groups.json"),
            json.dumps(asdict_without_nones(self.create_index_table())),
        )

        # Write out a separate JSON for each group
        groups_path = os.path.join(self.run_suite_path, "groups")
        ensure_directory_exists(groups_path)
        for group in self.schema.scenario_groups:
            tables = []
            table_names = []

            # Show the table aggregating over all the scenarios in this group (e.g., babi-1, babi-2, etc.)
            if len(self.group_scenario_model_to_runs[group.name]) > 1:
                table = self.create_group_table(
                    title=f"{group.display_name}",
                    scenario_group=group,
                    model_to_runs=self.group_model_to_runs[group.name],
                    link_to_runs=False,
                )
                tables.append(table)
                table_names.append(group.name)

            # Show the table per scenario
            for scenario in self.group_scenario_model_to_runs[group.name]:
                scenario_display_name = f"{group.display_name} / {scenario}"
                table = self.create_group_table(
                    title=scenario_display_name,
                    scenario_group=group,
                    model_to_runs=self.group_scenario_model_to_runs[group.name][scenario],
                    link_to_runs=True,
                )
                tables.append(table)
                table_names.append(group.name + "_" + scenario.replace(" ", ""))

            if len(tables) == 0:
                continue

            # Output latex file for each table
            # Add the latex_path to each table (changes `tables`!)
            base_path = os.path.join(groups_path, "latex")
            ensure_directory_exists(base_path)
            for table, name in zip(tables, table_names):
                latex_path = os.path.join(base_path, name + ".tex")
                table.links.append(Hyperlink(text="latex", href=latex_path))
                write(latex_path, table_to_latex(table, name))

            # Write JSON file
            write(
                os.path.join(groups_path, group.name + ".json"), json.dumps(list(map(asdict_without_nones, tables))),
            )


@htrack(None)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite", type=str, help="Name of the suite this run belongs to (default is today's date).", required=True,
    )
    args = parser.parse_args()

    # Output JSON files summarizing the benchmark results which will be loaded in the web interface
    summarizer = Summarizer(run_suite_path=os.path.join(args.output_path, "runs", args.suite))
    summarizer.read_runs()
    summarizer.write_models()
    summarizer.write_runs()
    summarizer.write_groups()
    summarizer.check_metrics_defined()
