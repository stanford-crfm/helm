import argparse
from dataclasses import dataclass, replace
import os
from collections import defaultdict
import urllib.parse

import dacite
from typing import List, Optional, Dict, Any, Tuple
import json

from common.general import write, ensure_directory_exists, asdict_without_nones, singleton, without_common_entries
from common.hierarchical_logger import hlog, htrack
from benchmark.scenarios.scenario import ScenarioSpec
from benchmark.adapter import AdapterSpec
from benchmark.metrics.statistic import Stat
from benchmark.runner import RunSpec
from proxy.models import ALL_MODELS, get_model
from .table import Cell, Table, Hyperlink, table_to_latex
from .schema import MetricNameMatcher, RunGroup, read_schema, SCHEMA_YAML_PATH
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


def dict_to_str(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def get_scenario_name(group: RunGroup, scenario_spec: ScenarioSpec):
    return group.name + "_" + dict_to_str(scenario_spec.args).replace(" ", "").replace("/", "_")


def get_scenario_display_name(group: RunGroup, scenario_spec: ScenarioSpec):
    return f"{group.display_name} / {dict_to_str(scenario_spec.args)}"


class Summarizer:
    """Summarize the benchmark results in JSON files to be displayed in the UI."""

    def __init__(self, suite: str, output_path: str):
        self.suite: str = suite
        self.run_suite_path: str = os.path.join(output_path, "runs", suite)

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

        # For each scenario group (e.g., natural_qa), map
        # (i) scenario spec (e.g., subject=philosophy) [optional] and
        # (ii) adapter spec (e.g., model = openai/davinci)
        # to list of runs
        self.group_adapter_to_runs: Dict[str, Dict[AdapterSpec, List[Run]]] = defaultdict(lambda: defaultdict(list))
        self.group_scenario_adapter_to_runs: Dict[ScenarioSpec, Dict[AdapterSpec, List[Run]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for run in self.runs:
            for group in run.run_spec.groups:
                scenario_spec = run.run_spec.scenario_spec
                adapter_spec = run.run_spec.adapter_spec

                self.group_adapter_to_runs[group][adapter_spec].append(run)
                self.group_scenario_adapter_to_runs[group][scenario_spec][adapter_spec].append(run)

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

    @htrack(None)
    def expand_subgroups(self, group: RunGroup) -> List[RunGroup]:
        """Given a RunGroup, collect a list of its subgroups by traversing the subgroup tree."""
        return [group] + [
            subsubgroup
            for subgroup in group.subgroups
            for subsubgroup in self.expand_subgroups(self.schema.name_to_run_group[subgroup])
        ]

    def create_index_tables(self) -> Table:
        category_to_groups = defaultdict(list)
        for group in self.schema.run_groups:
            category_to_groups[group.category].append(group)

        tables = []
        for category, groups in category_to_groups.items():
            # create group tables
            header = [
                Cell("Group"),
                Cell("Description"),
                Cell("# models"),
            ]
            rows = []
            for group in groups:
                num_models = len(
                    set(
                        adapter_spec.model
                        for subgroup in self.expand_subgroups(group)
                        for adapter_spec in self.group_adapter_to_runs[subgroup.name].keys()
                    )
                )
                if num_models == 0:
                    continue
                rows.append(
                    [
                        Cell(group.display_name, href=get_benchmarking_url({"suite": self.suite, "group": group.name})),
                        Cell(group.description),
                        Cell(num_models),
                    ]
                )
            tables.append(Table(title=category, header=header, rows=rows))

        return tables

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
        self,
        title: str,
        run_group: RunGroup,
        adapter_to_runs: Dict[str, List[Run]],
        link_to_runs: bool,
        columns: List[Tuple[RunGroup, str]],  # run_group, metric_group
    ) -> Table:
        """
        Create a table for a run_group (natural_qa) where each row is an
        adapter (e.g,  model) and columns are constructed based on metrics.
        """

        # Figure out what the columns of the table are.
        # Create header (cells to display) and the list of metric name filters
        # (to pull out information later).
        header: List[Cell] = []
        matchers: List[MetricNameMatcher] = []
        group_names: List[str] = []  # for each column

        header.append(Cell("Adapter"))
        for run_subgroup, metric_group_name in columns:
            if metric_group_name not in run_subgroup.metric_groups:
                continue
            metric_group = self.schema.name_to_metric_group[metric_group_name]
            for metric in metric_group.metrics:
                matcher = metric.substitute(run_subgroup.environment)
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

                if run_subgroup.name != run_group.name:
                    header_name = f"{run_subgroup.get_short_display_name()} ({header_field.get_short_display_name()})"

                header.append(Cell(header_name, description=description))
                matchers.append(matcher)
                group_names.append(run_subgroup.name)

        def run_spec_names_to_url(run_spec_names: List[str]) -> str:
            # TODO: include display names
            return get_benchmarking_url(
                {
                    "suite": self.suite,
                    "runSpecs": json.dumps(run_spec_names),
                    "scenarioDisplayName": title,
                    "scenarioDescription": run_group.description,
                }
            )

        # Compute adapter names (TODO: unify with findDiff)
        infos = without_common_entries(list(map(asdict_without_nones, adapter_to_runs.keys())))

        # Populate the contents of the table
        rows = []
        for (adapter_spec, runs), info in zip(adapter_to_runs.items(), infos):
            model = get_model(adapter_spec.model)
            display_name = model.display_name + (f" [{dict_to_str(info)}]" if len(info) > 0 else "")

            # Link to all the runs under this model
            if link_to_runs:
                run_spec_names = [run.run_spec.name for run in runs]
                href = run_spec_names_to_url(run_spec_names)
            else:
                href = None

            # Render contamination information
            point = self.contamination.get_point(model.name, run_group.name)
            if point is not None:
                suffix = CONTAMINATION_SYMBOLS[point.level]  # Append to name of model
                description = point.description
                contamination_level = point.level
            else:
                suffix = ""
                description = ""
                contamination_level = None

            cells = [Cell(display_name + suffix, description=description, href=href)]
            for group_name, matcher in zip(group_names, matchers):
                group_runs = list(filter(lambda run: group_name in run.run_spec.groups, runs))
                cells.append(self.create_cell(group_runs, matcher, contamination_level))

            rows.append(cells)

        # Link to all runs under all models (to compare models)
        all_run_spec_names = []
        for runs in adapter_to_runs.values():
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
            json.dumps(list(map(asdict_without_nones, self.create_index_tables()))),
        )

        # Write out a separate JSON for each group
        groups_path = os.path.join(self.run_suite_path, "groups")
        ensure_directory_exists(groups_path)
        for group in self.schema.run_groups:
            tables = []
            table_names = []

            # Collect all subgroups
            subgroups = self.expand_subgroups(group)

            # If there are multiple subgroups, aggregate into a single adapter_to_run dict and organize per metric
            if len(subgroups) > 1:
                adapter_to_runs = defaultdict(list)
                all_metric_groups = []
                for subgroup in subgroups:
                    all_metric_groups.extend(subgroup.metric_groups)
                    for adapter_spec, runs in self.group_adapter_to_runs[subgroup.name].items():
                        # TODO: different scenarios might have different adapters (e.g., instructions) but we still want
                        # to visualize them in the same row, so for now we just keep the model part of the spec
                        adapter_spec = AdapterSpec(model=adapter_spec.model, method="none")
                        adapter_to_runs[adapter_spec].extend(runs)
                all_metric_groups = list(dict.fromkeys(all_metric_groups))  # deduplicate while preserving order

                if len(adapter_to_runs) > 0:
                    for metric_group in all_metric_groups:
                        display_name = self.schema.name_to_metric_group[metric_group].get_short_display_name()
                        table = self.create_group_table(
                            title=f"{display_name}",
                            run_group=group,
                            adapter_to_runs=adapter_to_runs,
                            columns=[(subgroup, metric_group) for subgroup in subgroups],
                            link_to_runs=False,
                        )
                        tables.append(table)
                        table_names.append(metric_group)
            else:
                # Create a table aggregating over all scenarios in each group
                for subgroup in subgroups:
                    if len(self.group_scenario_adapter_to_runs[subgroup.name]) > 1:
                        table = self.create_group_table(
                            title=f"{subgroup.display_name}",
                            run_group=subgroup,
                            adapter_to_runs=self.group_adapter_to_runs[subgroup.name],
                            columns=[(subgroup, metric_group) for metric_group in subgroup.metric_groups],
                            link_to_runs=False,
                        )
                        tables.append(table)
                        table_names.append(group.name)

                    # Show the table per scenario
                    for scenario_spec in self.group_scenario_adapter_to_runs[subgroup.name]:
                        scenario_name = get_scenario_name(subgroup, scenario_spec)
                        scenario_display_name = get_scenario_display_name(subgroup, scenario_spec)
                        table = self.create_group_table(
                            title=scenario_display_name,
                            run_group=subgroup,
                            adapter_to_runs=self.group_scenario_adapter_to_runs[subgroup.name][scenario_spec],
                            columns=[(subgroup, metric_group) for metric_group in subgroup.metric_groups],
                            link_to_runs=True,
                        )
                        tables.append(table)
                        table_names.append(scenario_name)

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
    summarizer = Summarizer(suite=args.suite, output_path=args.output_path)
    summarizer.read_runs()
    summarizer.write_models()
    summarizer.write_runs()
    summarizer.write_groups()
    summarizer.check_metrics_defined()
