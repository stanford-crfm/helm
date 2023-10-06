# mypy: check_untyped_defs = False
"""Reads the output of the benchmark runs and produces:
- JSON files for the frontend
- Tables for the paper

Usage:

    venv/bin/helm-summarize --suite <Name of the suite>
"""

import argparse
import cattrs
import os
import datetime
import urllib.parse
import json
import yaml
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import mean, median
from typing import List, Optional, Dict, Any, Tuple, Set

from tqdm import tqdm

from helm.common.general import (
    write,
    ensure_directory_exists,
    asdict_without_nones,
    parallel_map,
    singleton,
    unique_simplification,
)
from helm.common.codec import from_json
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.data_overlap.data_overlap_spec import DataOverlapStats, GroupOverlapStats
from helm.benchmark.data_overlap.light_scenario import ScenarioSpecInstanceIds
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric import get_all_stats_by_name
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.runner import RunSpec, LATEST_SYMLINK
from helm.benchmark.presentation.table import Cell, HeaderCell, Table, Hyperlink, table_to_latex
from helm.benchmark.presentation.schema import (
    MetricNameMatcher,
    RunGroup,
    read_schema,
    SCHEMA_YAML_FILENAME,
    BY_GROUP,
    THIS_GROUP_ONLY,
    NO_GROUPS,
)
from helm.benchmark.presentation.contamination import (
    read_contamination,
    validate_contamination,
    CONTAMINATION_SYMBOLS,
    CONTAMINATION_STYLES,
    CONTAMINATION_LEVEL_STRONG,
)
from helm.benchmark.presentation.run_display import write_run_display_json


OVERLAP_N_COUNT = 13


@dataclass(frozen=True)
class ExecutiveSummary:
    """
    Summary of the output of benchmarking.
    This is always loaded by the frontend, so keep this small.

    A note on the relation between `release`, `suites`, and `suite`:
    There are two modes for releasing runs: `release` and `suite`.
    `releases` contain a package of suites. When the `release` mode
    is used, `release` and `suites` will not be None and `suite`will be None.
    When `suite` mode is used, `suite` will not be None and `release`
    and `suites` will be None
    """

    release: Optional[str]
    suites: Optional[List[str]]
    suite: Optional[str]
    date: str

    # TODO: later, put model rankings, etc. here


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
        # HACK: if we are looking for `quasi_exact_match` and it's not there, try `exact_match` instead
        # This is necessary for prompting ablations at the moment, since some scenarios normally have quasi_exact_match
        # as the main metric but multiple_choice_separate_original only generates exact_match
        if matcher.name == "quasi_exact_match":
            hlog("WARNING: No quasi_exact_match metric found, looking for exact_match instead")
            matcher = replace(matcher, name="exact_match")
            matching_stats = [stat for stat in stats if matcher.matches(stat.name)]
            if len(matching_stats) == 0:
                return None
        else:
            return None

    # Matcher matches all sub splits so we should aggregate these
    if matcher.sub_split is None:
        stats_dict: Dict[MetricName, Stat] = {}
        for stat in matching_stats:
            stat = Stat(replace(stat.name, sub_split=None)).merge(stat)
            merge_stat(stats_dict, stat)
        matching_stats = list(stats_dict.values())

    return singleton(matching_stats)


def get_benchmarking_url(params: Dict[str, str]) -> str:
    # Don't encode ' ' as '+'
    return "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def dict_to_str(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def get_scenario_name(group: RunGroup, scenario_spec: ScenarioSpec):
    return group.name + "_" + dict_to_str(scenario_spec.args).replace(" ", "").replace("/", "_")


def get_coarse_adapter_spec(
    adapter_spec: AdapterSpec, scenario_spec: Optional[ScenarioSpec] = None, adapter_keys_shown: List[str] = []
) -> AdapterSpec:
    """
    Return an abstraction of an AdapterSpec that corresponds to the method
    (e.g., model, decoding parameters), and not the part that contains
    scenario-specific things like instructions.
    This is not an easy thing to disentangle, so just try our best
    in a necessarily scenario-specific way.
    """
    # TODO: clean up this logic a bit
    # Sometimes the instructions contain information about the scenario.
    if scenario_spec and scenario_spec.class_name.endswith(".MMLUScenario"):
        # MMLU: Sync up with logic in `get_mmlu_spec` for constructing the instructions.
        subject = scenario_spec.args["subject"].replace("_", " ")
        instructions = adapter_spec.instructions.replace(subject, "___")
    elif scenario_spec and scenario_spec.class_name.endswith(".RAFTScenario"):
        # RAFT scenario has arbitrary instructions, so impossible to remove
        # the scenario information, so remove all of it.
        instructions = "<scenario specific>"
    else:
        instructions = adapter_spec.instructions
    adapter_spec = replace(adapter_spec, instructions=instructions)

    # Create a new adapter_spec, keeping only the model and the keys in adapter_keys_shown
    adapter_spec_kwargs = {key: adapter_spec.__dict__[key] for key in adapter_keys_shown}
    return AdapterSpec(**adapter_spec_kwargs)  # type: ignore


def get_method_display_name(model_display_name: Optional[str], info: Dict[str, Any]) -> str:
    """
    Return a nice name to display for `adapter_spec` which denotes a method.
    `info` contains the decoding parameters.

    Format: Model (info...)
    """
    info = dict(info)
    if "model" in info:
        del info["model"]

    return (model_display_name or "???") + (f" [{dict_to_str(info)}]" if len(info) > 0 else "")


def compute_aggregate_row_win_rates(table: Table, aggregation: str = "mean") -> List[Optional[float]]:
    """
    Computes the aggregate win rate of each row across columns. For a given row r1 and column c1, the win rate of r1 wrt
    to c1 corresponds to: if we pick another row r2 uniformly at random, what is the probability that r1c1 is better
    that r2c1?
    `aggregation` determines how we aggregate win rates across columns, currently can be "mean" or "median".
    We skip columns where "better" is ambiguous or less than 2 values are non-null.
    Returns a list of aggregate win rates, one per row, with None if a row was never meaningfully comparable (i.e., all
    non-null values of the row are in columns we skip).
    """
    assert aggregation in ["mean", "median"]
    win_rates_per_row: List[List[float]] = [[] for _ in table.rows]
    for i, header_cell in enumerate(table.header):
        lower_is_better = header_cell.lower_is_better
        if lower_is_better is None:  # column does not have a meaningful ordering
            continue

        # sort row indices by cell value and then compute the number of wins as the index in the sorted list
        def is_cell_valid(cell: Cell) -> bool:  # ignore cells which are strongly contaminated or have no value
            if cell.value is None:
                return False
            if cell.contamination_level and cell.contamination_level == CONTAMINATION_LEVEL_STRONG:
                return False
            return True

        values = [(row[i].value, j) for j, row in enumerate(table.rows) if is_cell_valid(row[i])]
        if len(values) < 2:  # don't rank a single model
            continue
        for wins, (v, j) in enumerate(sorted(values, reverse=lower_is_better)):
            win_rate = wins / (len(values) - 1)  # normalize to [0, 1]
            win_rates_per_row[j].append(win_rate)

    # Note: the logic up to here is somewhat general as it simply computes win rates across columns for each row.
    # Here, we simply average these win rates but we might want some more involved later (e.g., weighted average).
    aggregate_win_rates: List[Optional[float]] = []
    for win_rates in win_rates_per_row:
        if len(win_rates) == 0:
            aggregate_win_rates.append(None)
        else:
            aggregate = mean(win_rates) if aggregation == "mean" else median(win_rates)
            aggregate_win_rates.append(aggregate)

    return aggregate_win_rates


AGGREGATE_WIN_RATE_COLUMN = 1


class Summarizer:
    """Summarize the benchmark results in JSON files to be displayed in the UI."""

    COST_REPORT_FIELDS: List[str] = ["num_prompt_tokens", "num_completion_tokens", "num_completions", "num_requests"]

    # We need to hide stats for these model-metric combinations
    LOGPROBS_ISSUE_MODELS: Set[str] = {
        "anthropic/stanford-online-all-v4-s3",
        # Together sometimes returns logprobs and sometimes does not.
        # TODO(#1847): Enabled calibration for metrics after this is resolved.
        "meta/llama-7b",
        "meta/llama-13b",
        "meta/llama-30b",
        "meta/llama-65b",
        "meta/llama-2-7b",
        "meta/llama-2-13b",
        "meta/llama-2-70b",
    }
    LOGPROBS_ISSUE_METRICS: Set[str] = {
        # MSMARCO metrics
        "NDCG@10",
        "RR@10",
        "NDCG@20",
        "RR@20",
        # Calibration metrics
        "ece_1_bin",
        "ece_10_bin",
        "platt_ece_1_bin",
        "platt_ece_10_bin",
        "platt_coef",
        "platt_intercept",
        "selective_cov_acc_area",
        "selective_acc@10",
    }

    def __init__(
        self,
        release: Optional[str],
        suites: Optional[List[str]],
        suite: Optional[str],
        output_path: str,
        verbose: bool,
        num_threads: int,
    ):
        """
        A note on the relation between `release`, `suites`, and `suite`:
        There are two modes for releasing runs: `release` and `suite`.
        `releases` contain a package of suites. When the `release` mode
        is used, `release` and `suites` will not be None and `suite` will be None.
        When `suite` mode is used, `suite` will not be None and `release`
        and `suites` will be None
        """
        # TODO(yifanmai): Delete the `suite` argument.
        self.output_path: str = output_path
        self.run_release_path: str
        self.suites: List[str]
        self.run_suite_paths: List[str]
        self.suite: Optional[str] = None
        self.release: Optional[str] = None
        if suite:
            self.suite = suite
            self.run_release_path = os.path.join(output_path, "runs", suite)
            self.run_suite_paths = [self.run_release_path]
            self.suites = [suite]
        elif release and suites:
            self.release = release
            self.suites = suites
            self.run_release_path = os.path.join(output_path, "releases", release)
            self.run_suite_paths = [os.path.join(output_path, "runs", suite) for suite in suites]
        self.verbose: bool = verbose
        self.num_threads: int = num_threads

        ensure_directory_exists(self.run_release_path)

        self.schema = read_schema()
        self.contamination = read_contamination()
        validate_contamination(self.contamination, self.schema)

    def read_run(self, run_path: str) -> Run:
        """Load the `Run` object from `run_path`."""

        with open(os.path.join(run_path, "run_spec.json")) as f:
            run_spec = from_json(f.read(), RunSpec)

        with open(os.path.join(run_path, "stats.json")) as f:
            stats = from_json(f.read(), List[Stat])

        return Run(
            run_path=run_path,
            run_spec=run_spec,
            stats=stats,
        )

    def filter_runs_by_visibility(self, runs: List[Run], group: RunGroup) -> List[Run]:
        """Filter the list of runs and only keep runs relevant to this group."""
        filtered_runs: List[Run] = []
        for run in runs:
            included = True
            if group.visibility == THIS_GROUP_ONLY:  # don't include the canonical runs when looking at, say, ablations
                included = False
            for run_group_name in run.run_spec.groups:  # go through the groups of the run to determine visibility
                if run_group_name not in self.schema.name_to_run_group:
                    hlog(
                        f"WARNING: group {run_group_name} mentioned in run spec {run.run_spec.name} "
                        f"but undefined in {SCHEMA_YAML_FILENAME}, skipping"
                    )
                    continue
                run_group = self.schema.name_to_run_group[run_group_name]
                if run_group.visibility == NO_GROUPS:  # this run should never be visible
                    included = False
                    break
                if run_group.visibility == THIS_GROUP_ONLY:  # this run is part of a group with partial visibility
                    if run_group.name == group.name:  # if this is the exact group we are visualizing, include for sure
                        included = True
                        break
                    else:  # we won't visualize unless we hit exactly the group with partial visibility
                        included = False
            if included:
                filtered_runs.append(run)
        return filtered_runs

    def read_runs_for_suite(self, suite, run_suite_path):
        """Load the runs in the run suite path."""
        # run_suite_path can contain subdirectories that are not runs (e.g. eval_cache, groups)
        # so filter them out.
        run_dir_names = sorted([p for p in os.listdir(run_suite_path) if p != "eval_cache" and p != "groups"])
        for run_dir_name in tqdm(run_dir_names, disable=None):
            run_spec_path: str = os.path.join(run_suite_path, run_dir_name, "run_spec.json")
            stats_path: str = os.path.join(run_suite_path, run_dir_name, "stats.json")
            if not os.path.exists(run_spec_path) or not os.path.exists(stats_path):
                hlog(f"WARNING: {run_dir_name} doesn't have run_spec.json or stats.json, skipping")
                continue
            run_path: str = os.path.join(run_suite_path, run_dir_name)
            self.runs.append(self.read_run(run_path))

        # For each group (e.g., natural_qa), map
        # (i) scenario spec (e.g., subject=philosophy) [optional] and
        # (ii) adapter spec (e.g., model = openai/davinci)
        # to list of runs
        for run in self.runs:
            if run.run_spec.name in self.runs_to_run_suites:
                hlog(
                    f"WARNING: Run entry {run.run_spec.name} is present in two different Run Suites. "
                    f"Defaulting to the latest assigned suite: {suite}"
                )
            self.runs_to_run_suites[run.run_spec.name] = suite

            scenario_spec = run.run_spec.scenario_spec
            adapter_spec = run.run_spec.adapter_spec
            for group_name in run.run_spec.groups:
                self.group_adapter_to_runs[group_name][adapter_spec].append(run)
                self.group_scenario_adapter_to_runs[group_name][scenario_spec][adapter_spec].append(run)

    def read_runs(self):
        self.runs: List[Run] = []
        self.runs_to_run_suites: Dict[str, str] = {}
        self.group_adapter_to_runs: Dict[str, Dict[AdapterSpec, List[Run]]] = defaultdict(lambda: defaultdict(list))
        self.group_scenario_adapter_to_runs: Dict[str, Dict[ScenarioSpec, Dict[AdapterSpec, List[Run]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for suite, run_suite_path in zip(self.suites, self.run_suite_paths):
            self.read_runs_for_suite(suite, run_suite_path)

    def read_overlap_stats(self):
        """
        Load the overlap stats in the run suite path.
        Concretely:
            - get group -> scenario_spec information from self.runs
                run_spec data
            - read the files in the data_overlap directory in run_suite_path
                which are scenario_spec -> overlap ids
            - get aggregate stats for group -> overlap ratio
        """

        def get_group_to_scenario_specs(run_specs: List[RunSpec]) -> Dict[str, List[ScenarioSpec]]:
            scenario_specs_to_groups: Dict[ScenarioSpec, List[str]] = {}
            for run_spec in run_specs:
                scenario_spec = run_spec.scenario_spec
                groups = run_spec.groups
                if (
                    scenario_spec.class_name
                    != "helm.benchmark.scenarios.synthetic_efficiency_scenario.SyntheticEfficiencyScenario"
                ):
                    scenario_specs_to_groups[scenario_spec] = groups

            group_to_scenario_specs: Dict[str, List[ScenarioSpec]] = {}
            for scenario_spec, groups in scenario_specs_to_groups.items():
                for group in groups:
                    if group not in group_to_scenario_specs:
                        group_to_scenario_specs[group] = []
                    group_to_scenario_specs[group].append(scenario_spec)
            return group_to_scenario_specs

        def get_stats_file_metadata(data_overlap_dir: str) -> Dict[str, List[str]]:
            """
            Takes the data_overlap_dir as input and returns a dictionary
            of stats_file_path -> List(model_names)

            Sample input:
            file_models_mapping:
            - file_name: file1
                model_names:
                - model1
                - model2
            - file_name: file2
                model_names:
                - model2
                - model3

            """
            metadata_file_path: str = os.path.join(data_overlap_dir, "metadata.yaml")
            if not os.path.exists(metadata_file_path):
                return {}

            with open(metadata_file_path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file)

            file_metadata: Dict[str, List[str]] = {}
            for entry in data["file_models_mapping"]:
                if "file_name" in entry and "model_names" in entry:
                    file_path: str = os.path.join(data_overlap_dir, entry["file_name"])
                    file_metadata[file_path] = entry["model_names"]

            return file_metadata

        self._model_group_overlap_stats: Dict[Tuple[str, str], GroupOverlapStats] = {}

        data_overlap_dir = os.path.join(self.run_release_path, "data_overlap")
        if not os.path.isdir(data_overlap_dir):
            hlog(f"Directory {data_overlap_dir} not found; skipped import of overlap results.")
            return

        group_to_scenario_specs = get_group_to_scenario_specs([run.run_spec for run in self.runs])

        stats_file_metadata = get_stats_file_metadata(data_overlap_dir)

        for file_path, model_names in stats_file_metadata.items():
            overlap_stats_jsons = open(file_path, "r").readlines()

            data_overlap_stats_list: List[DataOverlapStats] = []
            for overlap_stats_json in overlap_stats_jsons:
                overlap_stats_dict = json.loads(overlap_stats_json)
                data_overlap_stats_list.append(cattrs.structure(overlap_stats_dict, DataOverlapStats))

            scenario_spec_overlap_counts: Dict[ScenarioSpec, Tuple[int, int, int]] = {}
            for data_overlap_stats in data_overlap_stats_list:
                data_overlap_stats_key = data_overlap_stats.data_overlap_stats_key
                n = data_overlap_stats_key.overlap_protocol_spec.n
                if n == OVERLAP_N_COUNT:
                    light_scenario_key = data_overlap_stats_key.light_scenario_key
                    scenario_spec = light_scenario_key.scenario_spec
                    if scenario_spec in self.scenario_spec_instance_id_dict:
                        # Get statistics based on the subset of instance_ids that HELM uses for a scenario
                        instance_ids = self.scenario_spec_instance_id_dict[scenario_spec]
                        num_instances = len(instance_ids)
                        num_overlapping_inputs = len(
                            set(data_overlap_stats.instance_ids_with_overlapping_input) & set(instance_ids)
                        )
                        num_overlapping_references = len(
                            set(data_overlap_stats.instance_ids_with_overlapping_reference) & set(instance_ids)
                        )
                        scenario_spec_overlap_counts[scenario_spec] = (
                            num_instances,
                            num_overlapping_inputs,
                            num_overlapping_references,
                        )

            for group, scenario_specs in group_to_scenario_specs.items():
                group_num_instances = 0
                group_num_overlapping_inputs = 0
                group_num_overlapping_references = 0
                for scenario_spec in scenario_specs:
                    if scenario_spec in scenario_spec_overlap_counts:
                        (
                            num_instances,
                            num_overlapping_inputs,
                            num_overlapping_references,
                        ) = scenario_spec_overlap_counts[scenario_spec]
                        group_num_instances += num_instances
                        group_num_overlapping_inputs += num_overlapping_inputs
                        group_num_overlapping_references += num_overlapping_references
                if group_num_instances != 0:
                    group_overlap_stats = GroupOverlapStats(
                        group=group,
                        num_instances=group_num_instances,
                        num_overlapping_inputs=group_num_overlapping_inputs,
                        num_overlapping_references=group_num_overlapping_references,
                    )
                    for model_name in model_names:
                        # Assume model name will only be associated with single group overlap list for now
                        # can update to join lists if need arises
                        self._model_group_overlap_stats[(model_name, group)] = group_overlap_stats

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
                    f"WARNING: metric name {metric_name} undefined in {SCHEMA_YAML_FILENAME} "
                    f"but appears in {len(run_spec_names)} run specs, including {run_spec_names[0]}"
                )

    @htrack(None)
    def write_executive_summary(self):
        """Write the executive summary."""
        date = datetime.date.today().strftime("%Y-%m-%d")

        summary = ExecutiveSummary(
            release=self.release,
            suites=self.suites,
            suite=self.suite,
            date=date,
        )

        write(
            os.path.join(self.run_release_path, "summary.json"),
            json.dumps(asdict_without_nones(summary), indent=2),
        )

    @htrack(None)
    def write_cost_report(self):
        """Write out the information we need to calculate costs per model."""
        # TODO: move to write_executive_summary()
        models_to_costs: Dict[str, Dict[str]] = defaultdict(lambda: defaultdict(int))
        for run in self.runs:
            model: str = run.run_spec.adapter_spec.model

            for stat in run.stats:
                stat_name = stat.name.name
                if stat_name in Summarizer.COST_REPORT_FIELDS and not stat.name.split:
                    models_to_costs[model][stat_name] += stat.sum

        # Do a second pass to add up the total number of tokens
        for costs in models_to_costs.values():
            costs["total_tokens"] = costs["num_prompt_tokens"] + costs["num_completion_tokens"]

        write(
            os.path.join(self.run_release_path, "costs.json"),
            json.dumps(models_to_costs, indent=2),
        )

    def write_runs(self):
        write(
            os.path.join(self.run_release_path, "runs.json"),
            json.dumps(list(map(asdict_without_nones, self.runs)), indent=2),
        )

    def write_run_specs(self):
        write(
            os.path.join(self.run_release_path, "run_specs.json"),
            json.dumps(list(map(asdict_without_nones, [run.run_spec for run in self.runs])), indent=2),
        )

    def write_runs_to_run_suites(self):
        write(
            os.path.join(self.run_release_path, "runs_to_run_suites.json"),
            json.dumps(self.runs_to_run_suites, indent=2),
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

        def get_cell(stats: List[Stat], compute_mean: bool = False, compute_sum: bool = False) -> Cell:
            """Render a value."""
            if len(stats) == 0:
                return Cell(None)
            aggregate_stat = replace(stats[0])
            for stat in stats[1:]:
                aggregate_stat.merge(stat)
            if compute_mean:
                return Cell(aggregate_stat.mean, description=aggregate_stat.bare_str())
            if compute_sum:
                return Cell(aggregate_stat.sum, description=aggregate_stat.bare_str())
            raise Exception("Either specify compute_mean or compute_sum")

        tables: List[Table] = []
        for category, groups in category_to_groups.items():
            header = [
                HeaderCell("Group"),
                HeaderCell("Description"),
                # Synchronize these names with `schema.yaml`
                HeaderCell("Adaptation method", description="Adaptation strategy (e.g., generation)"),
                HeaderCell("# instances", description="Number of instances evaluated on"),
                HeaderCell("# references", description="Number of references provided per instance"),
                HeaderCell("# prompt tokens", description="Total number of prompt tokens"),
                HeaderCell("# completion tokens", description="Total number of completion tokens"),
                HeaderCell("# models", description="Number of models we're evaluating"),
            ]
            rows: List[List[Cell]] = []
            for group in groups:
                models: Set[str] = set()
                methods: Set[str] = set()
                num_instances: List[Stat] = []
                num_references: List[Stat] = []
                num_prompt_tokens: List[Stat] = []
                num_completion_tokens: List[Stat] = []

                # Go over all the matching runs
                for subgroup in self.expand_subgroups(group):
                    for adapter_spec, runs in self.group_adapter_to_runs[subgroup.name].items():
                        filtered_runs = self.filter_runs_by_visibility(runs, subgroup)
                        models.add(adapter_spec.model)
                        methods.add(adapter_spec.method)
                        for run in filtered_runs:
                            num_instances.extend(get_all_stats_by_name(run.stats, "num_instances"))
                            num_references.extend(get_all_stats_by_name(run.stats, "num_references"))
                            num_prompt_tokens.extend(get_all_stats_by_name(run.stats, "num_prompt_tokens"))
                            num_completion_tokens.extend(get_all_stats_by_name(run.stats, "num_completion_tokens"))

                if len(num_instances) == 0:
                    continue

                rows.append(
                    [
                        Cell(group.display_name, href=get_benchmarking_url({"group": group.name})),
                        Cell(group.description, markdown=True),
                        Cell(", ".join(methods)),
                        get_cell(num_instances, compute_mean=True),
                        get_cell(num_references, compute_mean=True),
                        get_cell(num_prompt_tokens, compute_sum=True),
                        get_cell(num_completion_tokens, compute_sum=True),
                        Cell(len(models)),
                    ]
                )
            tables.append(Table(title=category, header=header, rows=rows))

        return tables

    def create_groups_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Create a table for each RunGroup category, linking to the pages where each one is displayed.
        """
        metadata = {}
        for group in self.schema.run_groups:
            metadata[group.name] = {
                "display_name": group.display_name,
                "description": group.description,
                "taxonomy": group.taxonomy and asdict_without_nones(group.taxonomy),
            }
        return metadata

    def create_cell(
        self,
        runs: List[Run],
        matcher: MetricNameMatcher,
        contamination_level: Optional[str],
        additional_info: Optional[str],
        hide_value: bool = False,
    ) -> Cell:
        """
        Use the metric name identified by `matcher` to pull out the stats from
        `runs` and return a representation of the average.
        There are four cases:
        1. No matching runs
        2. Matching runs but no matching stats (maybe stat was named incorrectly)
        3. Matching runs, matching stats, but stats have count = 0, so mean is undefined
           (e.g., bias metric ran and computed 0/0)
        4. Matching runs, matching stats, stats with count > 0

        In the first three cases, the cell value is None, but the description distinguishes between these cases.
        """
        # No runs at all
        if len(runs) == 0:
            return Cell(value=None, description="No matching runs")

        aggregate_stat: Optional[Stat] = None
        aggregated_run_specs: List[str] = []  # keep track of which run_specs we aggregate into the cell for debugging

        for run in runs:
            stat = get_unique_stat_by_matcher(run.stats, matcher)
            if stat is None:
                # Print out near misses to provide a more informative warning
                near_misses = [stat for stat in run.stats if stat.name.name == matcher.name]
                hlog(
                    f"WARNING: run spec {run.run_spec.name} does not have any stat matched by {matcher}, "
                    f"{len(near_misses)} near misses matching just the name"
                )
                if len(near_misses) > 0:
                    with htrack_block("Near misses"):
                        for stat in near_misses:
                            hlog(stat.name)
                continue

            if aggregate_stat is None:
                aggregate_stat = replace(stat)  # Important: copy!
            else:
                assert stat is not None  # Make type-checking happy
                aggregate_stat.merge(stat)
            aggregated_run_specs.append(run.run_spec.name)

        if aggregate_stat is None:
            return Cell(value=None, description=f"{len(runs)} matching runs, but no matching metrics")

        # TODO: need to exclude contaminated numbers somehow
        value: Optional[float] = None if hide_value else aggregate_stat.mean
        description = aggregate_stat.bare_str()
        if additional_info:
            description += "\n" + additional_info
        if self.verbose:
            description += "\n-- ".join(["\nRun specs:", *aggregated_run_specs])

        style: Dict[str, Any] = {}
        if contamination_level is not None:
            style = CONTAMINATION_STYLES.get(contamination_level, style)

        return Cell(value=value, description=description, style=style, contamination_level=contamination_level)

    def create_group_table(
        self,
        name: str,
        title: str,
        adapter_to_runs: Dict[AdapterSpec, List[Run]],
        link_to_runs: bool,
        columns: List[Tuple[RunGroup, str]],  # run_group, metric_group
        sort_by_model_order: bool = True,
        sub_split: Optional[str] = None,
        bold_columns: bool = True,
        add_win_rate: bool = False,
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
        if not columns or not adapter_to_runs:
            hlog(f"WARNING: table {title}, has no rows or columns, leaving empty")
            return Table("empty", [], [])

        header: List[HeaderCell] = []
        matchers: List[MetricNameMatcher] = []
        group_names: List[str] = []  # for each column
        num_groups = len(set(run_group.name for run_group, _ in columns))  # number of unique groups, determines headers

        # Column headers
        header.append(HeaderCell("Model/adapter"))
        for run_group, metric_group_name in columns:
            # check if at least the basic version of a metric group is evaluated (e.g., "bias" for "bias_detailed")
            if metric_group_name.replace("_detailed", "") not in run_group.metric_groups:
                continue
            metric_group = self.schema.name_to_metric_group[metric_group_name]
            for metric in metric_group.metrics:
                matcher = metric.substitute(run_group.environment)
                if sub_split is not None:
                    matcher = replace(matcher, sub_split=sub_split)
                header_field = self.schema.name_to_metric.get(matcher.name)
                if header_field is None:
                    hlog(f"WARNING: metric name {matcher.name} undefined in {SCHEMA_YAML_FILENAME}, skipping")
                    continue
                metadata = {
                    "metric": header_field.get_short_display_name(),
                    "run_group": run_group.get_short_display_name(),
                }

                header_name = header_field.get_short_display_name()
                description = (run_group.description + "\n\n" if run_group.description is not None else "") + (
                    (header_field.display_name if header_field.display_name else header_field.name)
                    + ": "
                    + (header_field.description if header_field.description is not None else "")
                )

                if matcher.perturbation_name is not None:
                    perturbation_field = self.schema.name_to_perturbation[matcher.perturbation_name]
                    header_name += " (" + perturbation_field.get_short_display_name() + ")"
                    description += (
                        "\n- Perturbation "
                        + (perturbation_field.display_name or perturbation_field.name)
                        + ": "
                        + (perturbation_field.description or "???")
                    )
                    metadata["perturbation"] = perturbation_field.get_short_display_name()

                if num_groups > 1:  # we have multiple groups in the same table, so display the name in the column
                    header_name = f"{run_group.get_short_display_name()} - {header_name}"

                header.append(
                    HeaderCell(
                        header_name,
                        description=description,
                        lower_is_better=header_field.lower_is_better,
                        metadata=metadata,
                    )
                )
                matchers.append(matcher)
                group_names.append(run_group.name)

        # TODO: Fix run_group logic
        run_group = columns[0][0]

        def run_spec_names_to_url(run_spec_names: List[str]) -> str:
            return get_benchmarking_url(
                {
                    "group": run_group.name,
                    "subgroup": title,
                    "runSpecs": json.dumps(run_spec_names),
                }
            )

        adapter_specs: List[AdapterSpec] = list(adapter_to_runs.keys())
        if sort_by_model_order:
            # Sort models by the order defined in the schema.
            # Models not defined in the schema will be sorted alphabetically and
            # placed before models in defined the schema.
            model_order = [model.name for model in self.schema.models]

            def _adapter_spec_sort_key(spec):
                index = model_order.index(spec.model) if spec.model in model_order else -1
                return (index, spec.model)

            adapter_specs = list(sorted(adapter_specs, key=_adapter_spec_sort_key))

        # Pull out only the keys of the method adapter_spec that is needed to
        # uniquely identify the method.
        infos = unique_simplification(list(map(asdict_without_nones, adapter_specs)), ["model"])

        assert len(adapter_specs) == len(infos), [adapter_specs, infos]

        # Populate the contents of the table
        rows = []
        for adapter_spec, info in zip(adapter_specs, infos):
            model_name: str = adapter_spec.model

            # Get the model display name from the schema.
            # Fall back to using the model name as the model display name if the model is not
            # defined in the schema.
            model_display_name = (
                self.schema.name_to_model[model_name].display_name
                if model_name in self.schema.name_to_model
                else model_name
            )

            runs = adapter_to_runs[adapter_spec]
            display_name = get_method_display_name(model_display_name, info)

            # Link to all the runs under this model
            if link_to_runs:
                run_spec_names = [run.run_spec.name for run in runs]
                href = run_spec_names_to_url(run_spec_names)
            else:
                href = None

            # Render contamination information
            point = self.contamination.get_point(model_name, columns[0][0].name)
            if num_groups == 1 and point is not None:  # display contamination information at the adapter level
                cells = [
                    Cell(display_name + CONTAMINATION_SYMBOLS[point.level], description=point.description, href=href)
                ]
            else:
                cells = [Cell(display_name, description="", href=href)]
            assert len(group_names) == len(matchers)
            for group_name, matcher in zip(group_names, matchers):
                group_runs = [run for run in runs if group_name in run.run_spec.groups]
                # HACK: when looking at aggregate bAbi results (e.g., reasoning), we want to see only the `task: all`
                # version and not the default aggregation across a sparse set of tasks, e.g., `task: {all, 3, 15, 19}`
                if "babi" in group_name and "task:" not in name:
                    group_runs = [run for run in group_runs if "task=all" in run.run_spec.name]

                point = self.contamination.get_point(model_name, group_name)
                if point is not None:
                    description = CONTAMINATION_SYMBOLS[point.level] + " " + point.description
                    contamination_level = point.level
                else:
                    description = ""
                    contamination_level = None

                group_overlap_stats = None
                if (model_name, group_name) in self._model_group_overlap_stats:
                    group_overlap_stats = self._model_group_overlap_stats[(model_name, group_name)]

                    description = (
                        f"Overlapping input ratio: {group_overlap_stats.overlapping_input_ratio:.3f}\n"
                        f"Overlapping reference ratio: {group_overlap_stats.overlapping_reference_ratio:.3f}\n"
                        f"{description}"
                    )

                # HACK: we want to hide stats for the following model-metric combinations:
                # 1. Calibration metrics + AI21/Anthropic
                # 2. MSMARCO metrics + AI21/Anthropic
                hide_value: bool = (
                    model_name in Summarizer.LOGPROBS_ISSUE_MODELS and matcher.name in Summarizer.LOGPROBS_ISSUE_METRICS
                )
                cells.append(
                    self.create_cell(
                        group_runs,
                        matcher,
                        contamination_level,
                        additional_info=description,
                        hide_value=hide_value,
                    )
                )

            rows.append(cells)

        # Link to a page to visualize all runs for comparison.
        # There could be a ton of runs, so only do this if there are 2-5
        # TODO: replace in frontend with a selector to choose which rows to visualize.
        links = []
        if link_to_runs:
            all_run_spec_names = []
            for adapter_spec, runs in adapter_to_runs.items():
                if len(runs) > 1:
                    hlog(
                        f"WARNING: table row corresponding to adapter spec {adapter_spec} has {len(runs)} > 1 runs:"
                        f" {[run.run_spec.name for run in runs]}"
                    )
                for run in runs:
                    all_run_spec_names.append(run.run_spec.name)
            if len(all_run_spec_names) >= 2 and len(all_run_spec_names) <= 5:
                links.append(Hyperlink(text="compare all", href=run_spec_names_to_url(all_run_spec_names)))

        table = Table(title=title, header=header, rows=rows, links=links, name=name)

        if add_win_rate:
            # add overall win rate as the second column
            WIN_RATE_AGGREGATION = "mean"
            win_rates = compute_aggregate_row_win_rates(table, aggregation=WIN_RATE_AGGREGATION)
            description = "How many models this model outperform on average (over columns)."
            table.header.insert(
                AGGREGATE_WIN_RATE_COLUMN,
                HeaderCell(
                    f"{WIN_RATE_AGGREGATION.capitalize()} win rate",
                    description=description,
                    lower_is_better=False,
                ),
            )
            for row, win_rate in zip(table.rows, win_rates):
                row.insert(AGGREGATE_WIN_RATE_COLUMN, Cell(win_rate))

        if bold_columns:
            for i, header_cell in enumerate(table.header):
                lower_is_better = header_cell.lower_is_better
                if lower_is_better is None:
                    continue
                values = [float(row[i].value) for row in rows if row[i].value is not None]
                if not values:
                    continue
                best = min(values) if lower_is_better else max(values)
                for row in table.rows:
                    cell = row[i]
                    if cell.value is not None and cell.value == best:
                        bold_style = cell.style.copy() if cell.style is not None else {}
                        bold_style["font-weight"] = "bold"
                        row[i] = replace(cell, style=bold_style)
        return table

    def create_group_tables_by_metric_group(self, group: RunGroup) -> List[Table]:
        """Creates a list of tables, one for each metric group (e.g., accuracy, robustness).
        Each table has `adapter_spec`s as rows and scenarios or groups as columns."""
        tables: List[Table] = []
        adapter_to_runs: Dict[AdapterSpec, List[Run]] = defaultdict(list)
        subgroups = self.expand_subgroups(group)
        for subgroup in subgroups:
            for adapter_spec, runs in self.group_adapter_to_runs[subgroup.name].items():
                coarse_adapter_spec = get_coarse_adapter_spec(adapter_spec, adapter_keys_shown=group.adapter_keys_shown)
                filtered_runs = self.filter_runs_by_visibility(runs, group)
                if filtered_runs:
                    adapter_to_runs[coarse_adapter_spec].extend(filtered_runs)

        all_metric_groups: List[str] = []
        if group.metric_groups:  # if the group specifies the metric groups, use them
            all_metric_groups = group.metric_groups
        else:  # otherwise, collect them from subgroups
            for subgroup in subgroups:
                all_metric_groups.extend(subgroup.metric_groups)
            # deduplicate, remove basic metric group if we include the detailed one, remove hidden metric groups
            all_metric_groups = [
                metric_group
                for metric_group in dict.fromkeys(all_metric_groups)
                if f"{metric_group}_detailed" not in all_metric_groups
                and metric_group not in group.subgroup_metric_groups_hidden
            ]

        if len(adapter_to_runs) > 0:
            for metric_group in all_metric_groups:
                display_name = self.schema.name_to_metric_group[metric_group].get_short_display_name()
                table = self.create_group_table(
                    name=metric_group,
                    title=display_name,
                    adapter_to_runs=adapter_to_runs,
                    columns=[(subgroup, metric_group) for subgroup in subgroups],
                    link_to_runs=False,
                    add_win_rate=True,
                )
                tables.append(table)
        return tables

    def create_group_tables_by_subgroup(self, group: RunGroup) -> List[Table]:
        """Creates a list of tables, one for each subgroup (e.g., mmlu).
        Each table has coarsened `adapter_spec`s` as rows and metrics as columns."""
        all_tables: List[Table] = []
        subgroups = self.expand_subgroups(group)
        for subgroup in subgroups:
            tables: List[Table] = []
            scenarios_shown = 0
            columns = [(subgroup, metric_group) for metric_group in subgroup.metric_groups]
            # Show the table per scenario
            for scenario_spec in self.group_scenario_adapter_to_runs[subgroup.name]:
                scenario_name = get_scenario_name(subgroup, scenario_spec)
                scenario_display_name = dict_to_str(scenario_spec.args)
                if group.name not in scenario_name:
                    scenario_display_name = f"{subgroup.display_name} {scenario_display_name}"
                adapter_to_runs: Dict[AdapterSpec, List[Run]] = defaultdict(list)
                for adapter_spec, runs in self.group_scenario_adapter_to_runs[group.name][scenario_spec].items():
                    filtered_runs = self.filter_runs_by_visibility(runs, group)
                    coarse_adapter_spec = get_coarse_adapter_spec(adapter_spec, scenario_spec, group.adapter_keys_shown)
                    adapter_to_runs[coarse_adapter_spec].extend(filtered_runs)
                if adapter_to_runs and subgroup.metric_groups:
                    table = self.create_group_table(
                        title=scenario_display_name,
                        name=scenario_name,
                        adapter_to_runs=adapter_to_runs,
                        columns=columns,
                        link_to_runs=True,
                    )
                    tables.append(table)
                    scenarios_shown += 1

                    if subgroup.sub_splits is not None:
                        for sub_split in subgroup.sub_splits:
                            table = self.create_group_table(
                                title=f"{subgroup.display_name} (sub-split: {sub_split})",
                                name=f"{subgroup.name}:sub_split={sub_split}",
                                adapter_to_runs=adapter_to_runs,
                                columns=columns,
                                link_to_runs=False,
                                sub_split=sub_split,
                            )
                            tables.append(table)

            if scenarios_shown > 1:  # add aggregate table
                adapter_to_runs = defaultdict(list)
                for adapter_spec, runs in self.group_adapter_to_runs[subgroup.name].items():
                    filtered_runs = self.filter_runs_by_visibility(runs, group)
                    coarse_adapter_spec = get_coarse_adapter_spec(
                        adapter_spec,
                        adapter_keys_shown=group.adapter_keys_shown,
                    )
                    adapter_to_runs[coarse_adapter_spec].extend(filtered_runs)
                if adapter_to_runs and subgroup.metric_groups:
                    table = self.create_group_table(
                        title=str(subgroup.display_name),
                        name=subgroup.name,
                        adapter_to_runs=adapter_to_runs,
                        columns=columns,
                        link_to_runs=False,
                    )
                    tables = [table] + tables
            all_tables.extend(tables)

        return all_tables

    def write_groups(self):
        """
        Each group selects out a set of runs.

        For each group, output:
        - Main table (model x columns): each row aggregate over all runs that match the (group, model).
        - Table for each scenario spec.
        """

        # Write out index file with all the groups and basic stats
        write(
            os.path.join(self.run_release_path, "groups.json"),
            json.dumps(list(map(asdict_without_nones, self.create_index_tables())), indent=2),
        )

        # Write out metadata file for all groups
        write(
            os.path.join(self.run_release_path, "groups_metadata.json"),
            json.dumps(self.create_groups_metadata(), indent=2),
        )

        # Write out a separate JSON for each group
        groups_path = os.path.join(self.run_release_path, "groups")
        ensure_directory_exists(groups_path)
        for group in self.schema.run_groups:
            if group.subgroup_display_mode == BY_GROUP or len(self.expand_subgroups(group)) == 1:
                # Create table aggregating over all scenarios in each group and then expand each scenario (we always do
                # this when there are no additional subgroups)
                tables: List[Table] = self.create_group_tables_by_subgroup(group)
            else:
                # Create a table for each metric, showing one subgroup per column for each adapter
                tables: List[Table] = self.create_group_tables_by_metric_group(group)
            if len(tables) == 0:
                continue

            # Output latex and JSON file for each table
            # Add the latex and JSON path as links to each table (mutates the tables!)
            ensure_directory_exists(os.path.join(groups_path, "latex"))
            ensure_directory_exists(os.path.join(groups_path, "json"))
            for table in tables:
                latex_path = os.path.join(groups_path, "latex", f"{group.name}_{table.name}.tex")
                table.links.append(Hyperlink(text="LaTeX", href=latex_path))
                write(latex_path, table_to_latex(table, f"{table.name} ({group.name})"))

                json_path = os.path.join(groups_path, "json", f"{group.name}_{table.name}.json")
                table.links.append(Hyperlink(text="JSON", href=json_path))
                write(json_path, json.dumps(asdict_without_nones(table), indent=2))

            # Write master JSON file
            write(
                os.path.join(groups_path, group.name + ".json"),
                json.dumps(list(map(asdict_without_nones, tables)), indent=2),
            )

    def write_run_display_json(self, skip_completed: bool) -> None:
        def process(run: Run) -> None:
            write_run_display_json(run.run_path, run.run_spec, self.schema, skip_completed)

        parallel_map(process, self.runs, parallelism=self.num_threads)

    def read_scenario_spec_instance_ids(self, num_instances) -> None:
        """
        This file checks if there exists a file, scenario_spec_instance_ids.json
        that it can read the instance_ids associated with scenario_specs.

        It will write the num_instances used in the run as part of the file name

        If it doesn't exist, it will go through all the scenario_state files
        and parse the instance_ids and output it to the file for future uses

        Only when the scenario_specs for the data overlap script change
        (or num_instances are different), will this need to be rerun.

        In such cases, do not include the file as part of the data_overlap directory.
        """
        self.scenario_spec_instance_id_dict: Dict[ScenarioSpec, List[str]] = dict()

        data_overlap_dir = os.path.join(self.run_release_path, "data_overlap")
        if not os.path.isdir(data_overlap_dir):
            hlog(f"Directory {data_overlap_dir} not found; skipped producing instance ids file.")
            return

        scenario_spec_instance_ids_json = os.path.join(
            data_overlap_dir, f"scenario_spec_instance_ids_{num_instances}.jsonl"
        )
        if not os.path.exists(scenario_spec_instance_ids_json):
            hlog(f"No scenario spec instance ids json, writing to {scenario_spec_instance_ids_json}")
            self.write_scenario_spec_instance_ids_json(scenario_spec_instance_ids_json)
        else:
            hlog(f"Reading scenario spec instance ids json from {scenario_spec_instance_ids_json}")
            scenario_spec_instance_ids_jsons = open(scenario_spec_instance_ids_json, "r").readlines()

            for scenario_spec_instance_ids_json in scenario_spec_instance_ids_jsons:
                scenario_spec_instance_ids_dict = json.loads(scenario_spec_instance_ids_json)
                scenario_spec_instance_ids = cattrs.structure(scenario_spec_instance_ids_dict, ScenarioSpecInstanceIds)
                self.scenario_spec_instance_id_dict[
                    scenario_spec_instance_ids.scenario_spec
                ] = scenario_spec_instance_ids.instance_ids

    def write_scenario_spec_instance_ids_json(self, file_path) -> None:
        for run in self.runs:
            run_spec = run.run_spec
            scenario_spec = run_spec.scenario_spec
            if scenario_spec in self.scenario_spec_instance_id_dict:
                continue

            run_path = run.run_path
            instances_file_path = os.path.join(run_path, "instances.json")
            with open(instances_file_path, "r") as f:
                raw_instances = json.load(f)

            # Optimization: Don't structure to dataclass, since we only need to read `id`
            instance_ids = [raw_instance["id"] for raw_instance in raw_instances]
            self.scenario_spec_instance_id_dict[scenario_spec] = instance_ids

        all_scenario_spec_instance_ids = []
        for scenario_spec, instance_ids in self.scenario_spec_instance_id_dict.items():
            scenario_spec_instance_ids = ScenarioSpecInstanceIds(scenario_spec=scenario_spec, instance_ids=instance_ids)
            all_scenario_spec_instance_ids.append(scenario_spec_instance_ids)

        with open(file_path, "w") as f:
            f.writelines(
                f"{json.dumps(asdict_without_nones(scenario_spec_instance_ids))}\n"
                for scenario_spec_instance_ids in all_scenario_spec_instance_ids
            )

    def symlink_latest(self) -> None:
        # Create a symlink runs/latest -> runs/<name_of_suite>,
        # so runs/latest always points to the latest run suite.
        releases_dir: str = os.path.dirname(self.run_release_path)
        symlink_path: str = os.path.abspath(os.path.join(releases_dir, LATEST_SYMLINK))
        hlog(f"Symlinking {self.run_release_path} to {LATEST_SYMLINK}.")
        if os.path.islink(symlink_path):
            # Remove the previous symlink if it exists.
            os.unlink(symlink_path)
        os.symlink(os.path.abspath(self.run_release_path), symlink_path)

    def run_pipeline(self, skip_completed: bool, num_instances: int) -> None:
        """Run the entire summarization pipeline pipeline."""
        self.read_runs()
        self.check_metrics_defined()

        self.write_run_display_json(skip_completed)

        # Must happen after summarizer.write_run_display_json()
        # because it uses instances.json files
        self.read_scenario_spec_instance_ids(num_instances)

        # Must happen after summarizer.read_scenario_spec_instance_ids()
        # because it uses self.scenario_spec_instance_id_dict
        self.read_overlap_stats()

        self.write_executive_summary()
        self.write_runs()
        self.write_run_specs()
        self.write_runs_to_run_suites()
        self.write_groups()
        self.write_cost_report()

        self.symlink_latest()


@htrack(None)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this summarization should go under.",
    )
    parser.add_argument(
        "--release",
        type=str,
        help="Experimental: Name of the release this summarization should go under.",
    )
    parser.add_argument(
        "--suites", type=str, nargs="+", help="Experimental: List of suites to summarize for this this release."
    )
    parser.add_argument("-n", "--num-threads", type=int, help="Max number of threads used to summarize", default=8)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debugging information.",
    )
    parser.add_argument(
        "--skip-completed-run-display-json",
        action="store_true",
        help="Skip write_run_display_json() for runs which already have all output display JSON files",
    )
    parser.add_argument(
        "-num-instances",
        type=int,
        help="Number of instance ids we're using; only for annotating scenario spec instance ids file",
        default=1000,
    )
    args = parser.parse_args()

    release: Optional[str] = None
    suites: Optional[str] = None
    suite: Optional[str] = None
    if args.suite and (args.release or args.suites):
        raise ValueError("If --suite is specified, then --release and --suites must NOT be specified.")
    elif args.suite:
        # Comment this out while we have a trial period for the `release` method.
        # hlog(
        #     "WARNING: The --suite flag is deprecated. Using --release and --suites is now preferred, "
        #     "where --release specifies the name of a release and --suites specifies several run suites "
        #     "to be included in that release."
        # )
        suite = args.suite
    elif args.release or args.suites:
        if not args.release or not args.suites:
            raise ValueError("If --release is specified, then --suites must also be specified and vice versa")
        release = args.release
        suites = args.suites
    else:
        raise ValueError("Exactly one of --release or --suite must be specified.")

    # Output JSON files summarizing the benchmark results which will be loaded in the web interface
    summarizer = Summarizer(
        release=release,
        suites=suites,
        suite=suite,
        output_path=args.output_path,
        verbose=args.debug,
        num_threads=args.num_threads,
    )
    summarizer.run_pipeline(skip_completed=args.skip_completed_run_display_json, num_instances=args.num_instances)
    hlog("Done.")


if __name__ == "__main__":
    main()
