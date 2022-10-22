import argparse
from dataclasses import dataclass, replace
import os
from collections import defaultdict
import urllib.parse

import dacite
from typing import List, Optional, Dict, Any, Tuple, Set
import json

from common.general import write, ensure_directory_exists, asdict_without_nones, singleton, unique_simplification
from common.hierarchical_logger import hlog, htrack
from benchmark.scenarios.scenario import ScenarioSpec
from benchmark.adapter import AdapterSpec
from benchmark.metrics.statistic import Stat
from benchmark.runner import RunSpec
from proxy.models import ALL_MODELS, Model, get_model
from .table import Cell, Table, Hyperlink, table_to_latex
from .schema import (
    MetricNameMatcher,
    RunGroup,
    read_schema,
    SCHEMA_YAML_PATH,
    BY_GROUP,
    ALL_GROUPS,
    THIS_GROUP_ONLY,
    NO_GROUPS,
)
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
    """Represents a run with spec and stats."""

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
    return "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def dict_to_str(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def get_scenario_name(group: RunGroup, scenario_spec: ScenarioSpec):
    return group.name + "_" + dict_to_str(scenario_spec.args).replace(" ", "").replace("/", "_")


def get_method_adapter_spec(adapter_spec: AdapterSpec, scenario_spec: ScenarioSpec) -> AdapterSpec:
    """
    Return an abstraction of an AdapterSpec that corresponds to the method
    (e.g., model, decoding parameters), and not the part that contains
    scenario-specific things like instructions.
    This is not an easy thing to disentangle, so just try our best
    in a necessarily scenario-specific way.
    """
    # Sometimes the instructions contain information about the scenario.
    if scenario_spec.class_name.endswith(".MMLUScenario"):
        # MMLU: Sync up with logic in `get_mmlu_spec` for constructing the instructions.
        subject = scenario_spec.args["subject"].replace("_", " ")
        instructions = adapter_spec.instructions.replace(subject, "___")
    elif scenario_spec.class_name.endswith(".RAFTScenario"):
        # RAFT scenario has arbitrary instructions, so impossible to remove
        # the scenario information, so remove all of it.
        instructions = ""
    else:
        instructions = adapter_spec.instructions
    return replace(adapter_spec, instructions=instructions)


def get_method_display_name(model: Model, info: Dict[str, Any]) -> str:
    """
    Return a nice name to display for `adapter_spec` which denotes a method.
    `info` contains the decoding parameters.

    Format: Model (info...)
    """
    info = dict(info)
    if "model" in info:
        del info["model"]

    return model.display_name + (f" [{dict_to_str(info)}]" if len(info) > 0 else "")


class Summarizer:
    """Summarize the benchmark results in JSON files to be displayed in the UI."""

    COST_REPORT_FIELDS: List[str] = ["num_prompt_tokens", "num_completion_tokens", "num_completions", "num_requests"]

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

        return Run(
            run_path=run_path,
            run_spec=run_spec,
            stats=stats,
        )

    def read_runs(self):
        """Load the corresponding runs for the run specs in run_specs.json."""

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
        self.group_scenario_adapter_to_runs: Dict[str, Dict[ScenarioSpec, Dict[AdapterSpec, List[Run]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for run in self.runs:
            scenario_spec = run.run_spec.scenario_spec
            adapter_spec = get_method_adapter_spec(run.run_spec.adapter_spec, scenario_spec)

            # organize the groups of the run by visibility in order to decide which groups we should add it to,
            # for each visibility value (ALL_GROUPS, NO_GROUPS, THIS_GROUP_ONLY) we collect the RunGroups with that
            # visibility in a list
            group_names_by_visibility: Dict[str, List[str]] = defaultdict(list)
            for group_name in run.run_spec.groups:
                if group_name not in self.schema.name_to_run_group:
                    hlog(f"WARNING: group {group_name} undefined in {SCHEMA_YAML_PATH}")
                    continue
                group = self.schema.name_to_run_group[group_name]
                group_names_by_visibility[group.visibility].append(group_name)

            if len(group_names_by_visibility[NO_GROUPS]) > 0:
                continue  # this run is part of a hidden group, skip
            elif group_names_by_visibility[THIS_GROUP_ONLY]:  # if it is part of a group with THIS_GROUP_ONLY visibility
                relevant_group_names = group_names_by_visibility[THIS_GROUP_ONLY]  # add it to these groups only
            else:
                relevant_group_names = group_names_by_visibility[ALL_GROUPS]  # otherwise add it everywhere

            for group_name in relevant_group_names:
                self.group_adapter_to_runs[group_name][adapter_spec].append(run)
                self.group_scenario_adapter_to_runs[group_name][scenario_spec][adapter_spec].append(run)

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

    @htrack(None)
    def write_cost_report(self):
        """Write out the information we need to calculate costs per model."""
        models_to_costs: Dict[str, Dict[str]] = defaultdict(lambda: defaultdict(int))
        for run in self.runs:
            model: str = run.run_spec.adapter_spec.model

            for stat in run.stats:
                stat_name = stat.name.name
                if stat_name in Summarizer.COST_REPORT_FIELDS and not stat.name.split:
                    models_to_costs[model][stat_name] += stat.sum
        write(
            os.path.join(self.run_suite_path, "costs.json"),
            json.dumps(models_to_costs, indent=2),
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

    def expand_subgroups(self, group: RunGroup) -> List[RunGroup]:
        """Given a RunGroup, collect a list of its subgroups by traversing the subgroup tree."""

        def expand_subgroups_(group: RunGroup, visited: Set[str]) -> List[RunGroup]:
            if group.name in visited:
                return []
            visited.add(group.name)
            return [group] + [
                subsubgroup
                for subgroup in group.subgroups
                for subsubgroup in expand_subgroups_(self.schema.name_to_run_group[subgroup], visited)
            ]

        return expand_subgroups_(group, visited=set())

    def create_index_tables(self) -> List[Table]:
        """
        Create a table for each RunGroup category, linking to the pages where each one is displayed.
        """
        category_to_groups = defaultdict(list)
        for group in self.schema.run_groups:
            category_to_groups[group.category].append(group)

        tables: List[Table] = []
        for category, groups in category_to_groups.items():
            header = [
                Cell("Group"),
                Cell("Description"),
                Cell("# models"),
            ]
            rows: List[List[Cell]] = []
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

    def create_groups_metadata(self) -> List[Table]:
        """
        Create a table for each RunGroup category, linking to the pages where each one is displayed.
        """
        metadata = {}
        for group in self.schema.run_groups:
            metadata[group.name] = {
                "displayName": group.display_name,
                "description": group.description,
            }
        return metadata

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
        adapter_to_runs: Dict[AdapterSpec, List[Run]],
        link_to_runs: bool,
        columns: List[Tuple[RunGroup, str]],  # run_group, metric_group
        sort_by_model_order: bool = True,
    ) -> Table:
        """
        Create a table for where each row is an adapter (for which we have a set of runs) and columns are pairs of
        run_group (natural_qa) and metrics (accuracy). This method can be used to either create a table with multiple
        metrics for a single scenario or a table with multiple scenarios together.
        adapter (e.g,  model) and columns are constructed based on metrics.
        """

        # Figure out what the columns of the table are.
        # Create header (cells to display) and the list of metric name filters
        # (to pull out information later).
        if not columns:
            return Table("empty", [], [])

        header: List[Cell] = []
        matchers: List[MetricNameMatcher] = []
        group_names: List[str] = []  # for each column
        num_groups = len(set(run_group.name for run_group, _ in columns))  # number of unique groups, determines headers

        header.append(Cell("Adapter"))
        for run_group, metric_group_name in columns:
            if metric_group_name not in run_group.metric_groups:
                continue
            metric_group = self.schema.name_to_metric_group[metric_group_name]
            for metric in metric_group.metrics:
                matcher = metric.substitute(run_group.environment)
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

                if num_groups > 1:  # we have multiple groups in the same table, so display the name in the column
                    header_name = f"{run_group.get_short_display_name()} ({header_field.get_short_display_name()})"

                header.append(Cell(header_name, description=description))
                matchers.append(matcher)
                group_names.append(run_group.name)

        # TODO: Fix run_group logic
        run_group = columns[0][0]

        def run_spec_names_to_url(run_spec_names: List[str]) -> str:
            return get_benchmarking_url(
                {
                    "suite": self.suite,
                    "group": run_group.name,
                    "subgroup": title,
                    "runSpecs": json.dumps(run_spec_names),
                }
            )

        adapter_specs = adapter_to_runs.keys()
        if sort_by_model_order:
            model_order = [model.name for model in ALL_MODELS]
            adapter_specs = sorted(adapter_specs, key=lambda spec: model_order.index(spec.model))

        # Pull out only the keys of the method adapter_spec that is needed to
        # uniquely identify the method.
        infos = unique_simplification(list(map(asdict_without_nones, adapter_specs)), ["model"])

        # Populate the contents of the table
        rows = []
        for adapter_spec, info in zip(adapter_specs, infos):
            model = get_model(adapter_spec.model)
            runs = adapter_to_runs[adapter_spec]
            display_name = get_method_display_name(model, info)

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
                group_runs = [run for run in runs if group_name in run.run_spec.groups]
                cells.append(self.create_cell(group_runs, matcher, contamination_level))

            rows.append(cells)

        # Link to all runs under all models (to compare models)
        all_run_spec_names = []
        for runs in adapter_to_runs.values():
            for run in runs:
                all_run_spec_names.append(run.run_spec.name)
        links = [Hyperlink(text="all models", href=run_spec_names_to_url(all_run_spec_names))]

        return Table(title=title, header=header, rows=rows, links=links)

    def create_group_tables_by_metric_group(self, groups: List[RunGroup]) -> Dict[str, Table]:
        tables: Dict[str, Table] = {}
        adapter_to_runs: Dict[AdapterSpec, List[Run]] = defaultdict(list)
        all_metric_groups: List[str] = []
        for group in groups:
            all_metric_groups.extend(group.metric_groups)
            for adapter_spec, runs in self.group_adapter_to_runs[group.name].items():
                adapter_spec = get_method_adapter_spec(adapter_spec, runs[0].run_spec.scenario_spec)
                adapter_to_runs[adapter_spec].extend(runs)
        all_metric_groups = list(dict.fromkeys(all_metric_groups))  # deduplicate while preserving order

        if len(adapter_to_runs) > 0:
            for metric_group in all_metric_groups:
                display_name = self.schema.name_to_metric_group[metric_group].get_short_display_name()
                table = self.create_group_table(
                    title=display_name,
                    adapter_to_runs=adapter_to_runs,
                    columns=[(group, metric_group) for group in groups],
                    link_to_runs=False,
                )
                tables[metric_group] = table
        return tables

    def create_group_tables_by_scenario(self, group: RunGroup) -> Dict[str, Table]:
        tables: Dict[str, Table] = {}
        if len(self.group_scenario_adapter_to_runs[group.name]) > 1:
            table = self.create_group_table(
                title=group.display_name,
                adapter_to_runs=self.group_adapter_to_runs[group.name],
                columns=[(group, metric_group) for metric_group in group.metric_groups],
                link_to_runs=False,
            )
            tables[group.name] = table

        # Show the table per scenario
        for scenario_spec in self.group_scenario_adapter_to_runs[group.name]:
            scenario_name = get_scenario_name(group, scenario_spec)
            scenario_display_name = dict_to_str(scenario_spec.args)
            table = self.create_group_table(
                title=scenario_display_name,
                adapter_to_runs=self.group_scenario_adapter_to_runs[group.name][scenario_spec],
                columns=[(group, metric_group) for metric_group in group.metric_groups],
                link_to_runs=True,
            )
            tables[scenario_name] = table

        return tables

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

        # Write out metadata file for all groups
        write(
            os.path.join(self.run_suite_path, "groups_metadata.json"),
            json.dumps(self.create_groups_metadata()),
        )

        # Write out a separate JSON for each group
        groups_path = os.path.join(self.run_suite_path, "groups")
        ensure_directory_exists(groups_path)
        for group in self.schema.run_groups:
            tables = []
            table_names = []

            # Collect all subgroups, by expanding recursively; intermediate nodes in the implicit subgroup tree will not
            # get visualized if they don't have their own metric_groups
            subgroups = self.expand_subgroups(group)

            if group.subgroup_display_mode == BY_GROUP or len(subgroups) == 1:
                # Create table aggregating over all scenarios in each group and then expand each scenario (we always do
                # this when there are no additional subgroups)
                for subgroup in subgroups:
                    for table_name, table in self.create_group_tables_by_scenario(subgroup).items():
                        tables.append(table)
                        table_names.append(table_name)
            else:
                # Create a table for each metric, showing one subgroup per column for each adapter
                for table_name, table in self.create_group_tables_by_metric_group(subgroups).items():
                    tables.append(table)
                    table_names.append(table_name)
            if len(tables) == 0:
                continue

            # Output latex file for each table
            # Add the latex_path to each table (changes `tables`!)
            base_path = os.path.join(groups_path, "latex")
            ensure_directory_exists(base_path)
            for table, table_name in zip(tables, table_names):
                latex_path = os.path.join(base_path, f"{group.name}_{table_name}.tex")
                table.links.append(Hyperlink(text="latex", href=latex_path))
                write(latex_path, table_to_latex(table, f"{table_name} ({group.name})"))

            # Write JSON file
            write(
                os.path.join(groups_path, group.name + ".json"),
                json.dumps(list(map(asdict_without_nones, tables))),
            )


@htrack(None)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this run belongs to (default is today's date).",
        required=True,
    )
    args = parser.parse_args()

    # Output JSON files summarizing the benchmark results which will be loaded in the web interface
    summarizer = Summarizer(suite=args.suite, output_path=args.output_path)
    summarizer.read_runs()
    summarizer.write_models()
    summarizer.write_runs()
    summarizer.write_groups()
    summarizer.check_metrics_defined()
    summarizer.write_cost_report()


if __name__ == "__main__":
    main()
