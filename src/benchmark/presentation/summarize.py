import argparse
import dataclasses
from dataclasses import dataclass, field, replace
import os
import traceback
import yaml
from collections import defaultdict
from pathlib import Path
import urllib.parse

import dacite
from typing import List, Optional, Dict, Any, Set
import json

from benchmark.scenarios.scenario import TEST_SPLIT
from common.authentication import Authentication
from common.general import parse_hocon, write, ensure_directory_exists, asdict_without_nones, singleton
from common.hierarchical_logger import hlog, htrack
from benchmark.statistic import Stat
from benchmark.runner import RunSpec
from benchmark.metric_name import MetricName
from benchmark.metric import get_unique_stat_by_name
from proxy.models import ALL_MODELS, get_model

"""
Reads the output of the benchmark runs and produces:
- JSON files for the frontend
- Tables for the paper

Usage:

    venv/bin/benchmark-summarize

"""

SCHEMA_YAML_PATH: str = "src/proxy/static/schema.yaml"


@dataclass(frozen=True)
class Run:
    """ Represents a run with spec and stats. """

    # Directory name of the run (used by frontend to find the actual instances to load)
    run_path: str

    # Run spec for the run
    run_spec: RunSpec

    # Statistics for the run
    stats: List[Stat]


@dataclass(frozen=True)
class Field:
    """
    Represents a field in a specification (e.g., temperature for `AdapterSpec`
    or exact_match for `Metric` or `Typos` for perturbations).
    """
    # Internal name (usually no spaces, etc.)
    name: str

    # What is displayed to the user
    display_name: Optional[str] = None

    # Description of the field
    description: Optional[str] = None


@dataclass(frozen=True)
class MetricGroup(Field):
    """
    Expands to a set of columns that correspond to a set of coherent metrics
    (e.g., all bias metrics).
    The columns are defined by the cross product of `stat_names` and
    `perturbation_names`.
    """
    # Which k value to use
    display_k: Optional[int] = None

    # If not specified, then use the group's default names
    stat_names: Optional[List[str]] = field(default_factory=list)

    # Which perturbations to show (e.g., typos)
    perturbation_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GroupDisplay:
    # Which data split to report numbers on
    split: str

    # What are the metric names that correspond to the default metric (e.g., accuracy)
    # TODO: rename to metric_names
    stat_names: List[str]


# TODO: rename to RunGroup and remove GroupDisplay
@dataclass(frozen=True)
class Group(Field):
    display: Optional[GroupDisplay] = None


@dataclass
class Schema:
    """Specifies information about what to display."""

    adapter: List[Field]
    metrics: List[Field]
    perturbations: List[Field]
    metric_groups: List[MetricGroup]
    groups: List[Group]

    def __post_init__(self):
        self.name_to_metric = {metric.name: metric for metric in self.metrics}
        self.name_to_perturbation = {perturbation.name: perturbation for perturbation in self.perturbations}

@dataclass(frozen=True)
class Cell:
    # Semantic value (that can be used for sorting)
    value: Any

    # Optionally, if we want to render things specially (floating points to 3 decimal points)
    display_value: Optional[str] = None

    # Detailed description if hover over the cell
    description: Optional[str] = None

    # If we click on the link for this cell, it takes us somewhere
    href: Optional[str] = None


@dataclass(frozen=True)
class Table:
    title: str
    header: List[Cell]
    rows: List[List[Cell]]

# TODO: remove this once we actually standardize things
def standardize(perturbation_name: Optional[str]) -> str:
    return perturbation_name or "identity"

@dataclass(frozen=True)
class MetricNameMatcher:
    """
    The schema file specifies information about what metrics we want to specify,
    but it doesn't specify full `MetricName`s.  Instead, it specifies enough
    information in a `MetricNameMatcher` to pull out the relevant
    `MetricName`s.
    """
    name: str
    k: Optional[int]
    split: Optional[str]
    perturbation_name: Optional[str]

    def matches(self, metric_name: MetricName) -> bool:
        return self.name == metric_name.name and \
            (self.k is None or self.k == metric_name.k) and \
            (self.split is None or self.split == metric_name.split) and \
            (self.perturbation_name is None or standardize(self.perturbation_name) == standardize(metric_name.perturbation and metric_name.perturbation.name))

def get_unique_stat_by_matcher(stats: List[Stat], matcher: MetricNameMatcher) -> Optional[Stat]:
    matching_stats = [stat for stat in stats if matcher.matches(stat.name)]
    if len(matching_stats) == 0:
        return None
    return singleton(matching_stats)


def get_benchmarking_url(params: Dict[str, str]) -> str:
    return "benchmarking.html?" + urllib.parse.urlencode(params)


class Summarizer:
    """Summarize the benchmark results in JSON files to be displayed in the UI."""

    def __init__(self, run_suite_path: str):
        self.run_suite_path: str = run_suite_path

    def read_schema(self):
        hlog(f"Reading schema from {SCHEMA_YAML_PATH}...")
        with open(SCHEMA_YAML_PATH) as f:
            raw = yaml.safe_load(f)
            self.schema = dacite.from_dict(Schema, raw)

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

        # Build mapping from group (e.g., natural_qa) and model (e.g., openai/davinci) to list of runs
        self.group_model_to_runs: Dict[str, Dict[str, List[Run]]] = defaultdict(lambda : defaultdict(list))
        self.group_scenario_model_to_runs: Dict[str, Dict[str, List[Run]]] = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
        for run in self.runs:
            for group in run.run_spec.groups:
                model = run.run_spec.adapter_spec.model
                # TODO: make this less hacky (mmlu:subject=philosophy,model=openai_davinci) since this doesn't work for data_augmentation=...
                scenario = run.run_spec.name.split(",")[0]
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

        with open(SCHEMA_YAML_PATH) as f:
            metrics = yaml.safe_load(f)["metrics"]
            defined_metric_names = set(entry["name"] for entry in metrics)

        for metric_name, run_spec_names in metric_name_to_run_spec_names.items():
            if metric_name not in defined_metric_names:
                hlog(f"WARNING: metric name {metric_name} undefined in {SCHEMA_YAML_PATH} but appears in {len(run_spec_names)} run specs, including {run_spec_names[0]}")


    def write_models(self):
        write(
            os.path.join(self.run_suite_path, "models.json"),
            json.dumps(list(map(asdict_without_nones, ALL_MODELS)), indent=2),
        )

    def write_runs(self):
        # TODO: write more summary information and something smaller
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
        rows = [
            [
                Cell(value=group.name, display_value=group.display_name),
                Cell(group.description),
                Cell(len(self.group_model_to_runs[group.name])),
            ]
            for group in self.schema.groups
        ]
        return Table(
            title="Overview of results",
            header=header,
            rows=rows,
        )

    def create_cell(self, runs: List[Run], matcher: MetricNameMatcher) -> Cell:
        """Use the metric name identified by `matcher` to pull out the stats
        from `runs` and return a representation of the average."""
        if len(runs) == 0:
            return Cell(None)

        aggregate_stat: Optional[Stat] = None
        for run in runs:
            stat = get_unique_stat_by_matcher(run.stats, matcher)
            if stat is None:
                hlog(f"WARNING: {matcher} doesn\'t match {run.run_spec.name}")
                continue  # TODO: probably should make a note that stats are missing
            stat = stat.take_mean()  # Collapse to a single point

            if aggregate_stat is None:
                aggregate_stat = replace(stat)
            else:
                aggregate_stat.merge(stat)

        if aggregate_stat is None:
            return Cell(None)
        value = aggregate_stat.mean
        description = str(aggregate_stat)  # TODO: make this nicer
        return Cell(value=value, display_value=round(value, 3))


    def create_group_table(self, title: str, group: Group, model_to_runs: Dict[str, List[Run]], link_to_runs: bool) -> Table:
        """
        Create a table where each row is a model and columns are constructed based on the group
        """

        # Figure out what the columsn of the table are.
        # Create header (cells to display) and the list of metric name filters
        # (to pull out information later).
        header = []
        matchers = []
        for metric_group in self.schema.metric_groups:
            # Get stat names and perturbations
            k = metric_group.display_k
            split = group.display.split
            stat_names = metric_group.stat_names or group.display.stat_names
            perturbation_names = metric_group.perturbation_names

            for stat_name in stat_names:
                for perturbation_name in perturbation_names:
                    header_field = self.schema.name_to_metric[stat_name]
                    header_name = header_field.display_name
                    if perturbation_name != "identity":
                        header_name += " (" + self.schema.name_to_perturbation[perturbation_name].display_name + ")"

                    header.append(Cell(header_name, description=header_field.description))
                    matchers.append(MetricNameMatcher(name=stat_name, k=k, split=split, perturbation_name=perturbation_name))

        # Populate the contents of the table
        rows = []
        # TODO: need to average over scenarios (which themselves have a list of runs) instead of runs?
        # The two are the same if there are the same number of random seeds
        for model_name, runs in model_to_runs.items():
            model = get_model(model_name)
            # Link to all the runs under this model
            if link_to_runs:
                run_spec_names = [run.run_spec.name for run in runs]
                href = get_benchmarking_url({"runSpec": "|".join(run_spec_names)})
            else:
                href = None
            rows.append(
                [Cell(model.display_name, href=href)] +
                [self.create_cell(runs, matcher) for matcher in matchers]
            )

        return Table(
            title=title,
            header=header,
            rows=rows,
        )

    def write_groups(self):
        """
        Each group selcts out a set of runs.

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
        for group in self.schema.groups:
            tables = [
                self.create_group_table(title=group.name, group=group, model_to_runs=self.group_model_to_runs[group.name], link_to_runs=False)
            ] + [
                self.create_group_table(title=scenario, group=group, model_to_runs=self.group_scenario_model_to_runs[group.name][scenario], link_to_runs=True)
                for scenario in self.group_scenario_model_to_runs[group.name]
            ]
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
        default="latest",
    )
    args = parser.parse_args()

    # Output JSON files summarizing the benchmark results which will be loaded in the web interface
    summarizer = Summarizer(run_suite_path=os.path.join(args.output_path, "runs", args.suite))
    summarizer.read_schema()
    summarizer.read_runs()
    summarizer.write_models()
    summarizer.write_runs()
    summarizer.write_groups()
    summarizer.check_metrics_defined()
