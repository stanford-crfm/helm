import ast
import dataclasses
from dataclasses import dataclass, field
import json
from typing import List, Optional, Dict, Set, Tuple, FrozenSet
import dacite
from inspect import cleandoc
import mako.template
import yaml
import re
from enum import Enum
from importlib import resources

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import hlog
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.augmentations.perturbation_description import PERTURBATION_WORST
from helm.common.hierarchical_logger import hwarn


# TODO: change to `helm.benchmark.config`
SCHEMA_YAML_PACKAGE: str = "helm.benchmark.static"

# TODO: add heim, vhelm, etc.
SCHEMA_CLASSIC_YAML_FILENAME: str = "schema_classic.yaml"


_ADAPTER_SPEC_PACKAGE = "helm.benchmark.adaptation"
_ADAPTER_SPEC_FILENAME = "adapter_spec.py"
_ADAPTER_SPEC_CLASS_NAME = "AdapterSpec"

VALID_SPLITS: Set[str] = {"test", "valid", "__all__"}
TEMPLATE_VARIABLE_PATTERN = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")


@dataclass(frozen=True)
class Field:
    """
    Represents a field in a specification (e.g., `temperature` for the adapter,
    or `exact_match` for metrics or `typos` for perturbations.
    """

    # Internal name (usually no spaces, etc.)
    name: str

    # What is displayed to the user
    display_name: Optional[str] = None

    # What is displayed to the user (e.g., in a table header)
    short_display_name: Optional[str] = None

    # Description of the field
    description: Optional[str] = None

    # Whether a lower vaue for this field corresponds to a better model
    # (e.g., False for accuracy, True for perplexity, None for num_trials)
    lower_is_better: Optional[bool] = None

    def get_short_display_name(self) -> str:
        name = self.short_display_name or self.display_name or self.name
        return name


@dataclass(frozen=True)
class MetricNameMatcher:
    """
    The schema file specifies information about what metrics we want to specify,
    but it doesn't specify full `MetricName`s.  Instead, it specifies enough
    information in a `MetricNameMatcher` to pull out the relevant
    `MetricName`s.
    """

    # Name of the metric
    name: str

    # Which data split to report numbers on (e.g., TEST_SPLIT)
    split: str

    # Which sub split to report numbers on (e.g., toxic, non-toxic)
    # If None, will match all sub splits
    sub_split: Optional[str] = None

    # Which perturbation to show (e.g., robustness)
    perturbation_name: Optional[str] = None

    def matches(self, metric_name: MetricName) -> bool:
        if self.name != metric_name.name:
            return False

        if self.split != "__all__" and self.split != metric_name.split:
            return False

        # Optional
        if self.sub_split is not None and self.sub_split != metric_name.sub_split:
            return False

        metric_perturbation_name = metric_name.perturbation and metric_name.perturbation.name
        if self.perturbation_name != metric_perturbation_name:
            return False

        # If there is a perturbation, only return the worst
        if metric_name.perturbation and metric_name.perturbation.computed_on != PERTURBATION_WORST:
            return False

        return True

    def substitute(self, environment: Dict[str, str]) -> "MetricNameMatcher":
        return MetricNameMatcher(
            name=mako.template.Template(self.name).render(**environment),
            split=mako.template.Template(self.split).render(**environment),
            perturbation_name=(
                mako.template.Template(self.perturbation_name).render(**environment)
                if self.perturbation_name is not None
                else None
            ),
        )


@dataclass(frozen=True)
class MetricGroup(Field):
    """
    A list of metrics (which are presumably logically grouped).
    """

    metrics: List[MetricNameMatcher] = field(default_factory=list)

    hide_win_rates: Optional[bool] = None
    """If set to true, do not compute win rates."""

    aggregation_strategies: Optional[List[str]] = None
    """List with values in {'win_rate','mean'} that correspond to aggregations"""


BY_METRIC = "by_metric"
BY_GROUP = "by_group"
ALL_GROUPS = "all_groups"
THIS_GROUP_ONLY = "this_group_only"
NO_GROUPS = "no_groups"


@dataclass(frozen=True)
class RunGroup(Field):
    """
    Defines information about how a group of runs is displayed.
    """

    # What groups (by name) to show
    metric_groups: List[str] = field(default_factory=list)

    # Names of subgroups which this group contains. That is, when displaying this group, we will also (recursively)
    # display all of its subgroups. This allows us to aggregate runs in different ways without having to explicitly
    # annotate each run. For instance, we can display MMLU with the group "QA" by adding it as a subgroup.
    subgroups: List[str] = field(default_factory=list)

    # Names of subsplits that we want to visualize separately for this group
    sub_splits: Optional[List[str]] = None

    # When displaying the subgroups, should each table show all the metrics for each group (BY_GROUP) or show all the
    # groups for each metric (BY_METRIC)
    subgroup_display_mode: str = BY_METRIC

    # Any subgroup metric groups we want to hide (e.g., robustness when running without perturbations)
    subgroup_metric_groups_hidden: List[str] = field(default_factory=list)

    # Defines variables that are substituted in any of the metrics
    environment: Dict[str, str] = field(default_factory=dict)

    # Which category this group belongs to. These currently include "Scenarios", "Tasks", "Components" and are used to
    # clump different groups together for easier website navigation.
    category: str = "Scenarios"

    # Whether runs in this group should be displayed as part of other groups they belong to
    # If ALL_GROUPS (default), we include the run in all groups it belongs to
    # If NO_GROUPS, don't include any of this group's runs
    # If THIS_GROUP_ONLY, we include the run in this specific group but not to others (this is useful for ablations
    # where we want to display a run for the ablation group but not for the more general groups it belongs to).
    # Example: If a run has the groups ["imdb", "ablation_in_context"] and "imdb" has visibility ALL_GROUPS, while
    # "ablation_in_context" has visiblity THIS_GROUP_ONLY, then this run is displayed under "ablation_in_context", but
    # not under "imdb" (and thus is not aggregated with the canonical runs with groups ["imdb"].
    visibility: str = ALL_GROUPS

    # For scenarios
    taxonomy: Optional[TaxonomyInfo] = None

    # Whether we have yet to evaluate this model
    todo: bool = False

    # Which adapter_spec fields we should preserve when displaying methods for this group
    # When we are constructing a table where the rows are methods, what constitutes a "method" is given by the set of
    # adapter keys. By default, this should just be "model_deployment" (e.g., BLOOM), where details like
    # "num_train_instances" are "marginalized out". However, for ablations, we want to include both "model_deployment"
    # and "num_train_instances".
    # NOTE: "model" is kept for backward compatibility reason.
    # TODO: remove when we don't want helm-summarize to support runs before November 2023 anymore.
    adapter_keys_shown: List[str] = field(default_factory=lambda: ["model_deployment", "model"])

    # Optional short description of the run group.
    # This description is used in some space-constrained places in frontend tables.
    # If unset, the description field will be used instead.
    short_description: Optional[str] = None


@dataclass
class Schema:
    """Specifies information about what to display on the frontend."""

    # Information about each field
    metrics: List[Field] = field(default_factory=list)

    # Information about each perturbation
    perturbations: List[Field] = field(default_factory=list)

    # Group the metrics
    metric_groups: List[MetricGroup] = field(default_factory=list)

    # Group the scenarios
    run_groups: List[RunGroup] = field(default_factory=list)

    # Adapter fields (e.g., temperature)
    # Automatically populated from the docstrings in the AdapterSpec class definition.
    # Should not be specified in the user's YAML file.
    adapter: Optional[List[Field]] = None

    def __post_init__(self):
        self.name_to_metric = {metric.name: metric for metric in self.metrics}
        self.name_to_perturbation = {perturbation.name: perturbation for perturbation in self.perturbations}
        self.name_to_metric_group = {metric_group.name: metric_group for metric_group in self.metric_groups}
        self.name_to_run_group = {run_group.name: run_group for run_group in self.run_groups}


def get_adapter_fields() -> List[Field]:
    """Generate the adapter fields from the docstrings in the AdapterSpec class definition."""
    # Unfortunately there is no standard library support for getting docstrings of class fields,
    # so we have to do the parsing outselves. Fortunately, the parsing is quite straightforward.
    contents = resources.files(_ADAPTER_SPEC_PACKAGE).joinpath(_ADAPTER_SPEC_FILENAME).read_text()
    module_node = ast.parse(contents)
    adapter_spec_node = [
        node
        for node in ast.iter_child_nodes(module_node)
        if isinstance(node, ast.ClassDef) and node.name == _ADAPTER_SPEC_CLASS_NAME
    ][0]
    metadata_fields: List[Field] = []
    field_name: str = ""
    for node in ast.iter_child_nodes(adapter_spec_node):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # This node is a field definition.
            # Save the name of the field for later.
            field_name = node.target.id
        else:
            # If this is a docstring that immediately follows a field definition,
            # output an adapter field with the name set to  the field definition and
            # the description set to the docstring.
            if (
                field_name
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                description = cleandoc(node.value.value).replace("\n", " ")
                metadata_fields.append(Field(name=field_name, description=description))
            field_name = ""

    return metadata_fields


def get_default_schema_path() -> str:
    return str(resources.files(SCHEMA_YAML_PACKAGE).joinpath(SCHEMA_CLASSIC_YAML_FILENAME))


def read_schema(schema_path: str) -> Schema:
    hlog(f"Reading schema file {schema_path}...")
    with open(schema_path, "r") as f:
        if schema_path.endswith(".json"):
            raw = json.load(f)
        else:
            raw = yaml.safe_load(f)
    schema = dacite.from_dict(Schema, raw)
    if schema.adapter:
        hwarn(f"The `adapter` field is deprecated and should be removed from schema file {schema_path}")
    return dataclasses.replace(schema, adapter=get_adapter_fields())


class ValidationSeverity(str, Enum):
    """Severity level for validation messages."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class SchemaValidationMessage:
    """Represents a single validation issue found in a schema."""

    severity: ValidationSeverity
    message: str
    schema_path: Optional[str] = None
    location: Optional[str] = None

    def __str__(self) -> str:
        parts = []
        if self.schema_path:
            parts.append(f"[{self.schema_path}]")
        parts.append(f"[{self.severity.value.upper()}]")
        if self.location:
            parts.append(f"at {self.location}:")
        parts.append(self.message)
        return " ".join(parts)


class SchemaValidationError(ValueError):
    """Exception raised when schema validation fails with errors."""

    def __init__(self, messages: List[SchemaValidationMessage]):
        self.messages = messages
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        super().__init__(
            f"Schema validation failed with {len(error_messages)} error(s):\n"
            + "\n".join(str(msg) for msg in error_messages)
        )


def is_template_variable(value: str) -> bool:
    """Check if a value is a valid template variable of the form ${var_name}."""
    if not value:
        return False
    return bool(TEMPLATE_VARIABLE_PATTERN.match(value))


def _detect_cycles_in_subgroups(
    run_groups: List[RunGroup],
    name_to_run_group: Dict[str, RunGroup],
) -> List[Tuple[str, List[str]]]:
    """Detect all cycles in the subgroup graph using 3-color DFS."""
    UNVISITED, VISITING, VISITED = 0, 1, 2

    valid_run_groups = [rg for rg in run_groups if rg.name and rg.name.strip()]
    visit_status: Dict[str, int] = {rg.name: UNVISITED for rg in valid_run_groups}

    cycles: List[Tuple[str, List[str]]] = []
    reported_cycle_nodes: Set[FrozenSet[str]] = set()

    def dfs(node_name: str, path: List[str]) -> None:
        if not node_name or node_name not in visit_status:
            return

        if visit_status[node_name] == VISITING:
            if node_name in path:
                cycle_start_idx = path.index(node_name)
                cycle_path = path[cycle_start_idx:] + [node_name]
                cycle_nodes = frozenset(cycle_path[:-1])

                if cycle_nodes not in reported_cycle_nodes:
                    reported_cycle_nodes.add(cycle_nodes)
                    cycles.append((node_name, cycle_path))
            return

        if visit_status[node_name] == VISITED:
            return

        visit_status[node_name] = VISITING
        path.append(node_name)

        if node_name in name_to_run_group:
            run_group = name_to_run_group[node_name]
            subgroups = run_group.subgroups or []
            for subgroup_name in subgroups:
                dfs(subgroup_name, path)

        path.pop()
        visit_status[node_name] = VISITED

    for run_group in valid_run_groups:
        if visit_status.get(run_group.name) == UNVISITED:
            dfs(run_group.name, [])

    return cycles


def validate_schema(
    schema: Schema,
    *,
    schema_path: Optional[str] = None,
    strict: bool = True,
    check_parent_child_partition: bool = False,
    check_orphan_children: bool = False,
) -> List[SchemaValidationMessage]:
    """Validate a Schema object and return a list of validation messages.

    If strict=True, raises SchemaValidationError when validation errors are found.
    """
    messages: List[SchemaValidationMessage] = []

    def add_error(message: str, location: Optional[str] = None) -> None:
        messages.append(
            SchemaValidationMessage(
                severity=ValidationSeverity.ERROR,
                message=message,
                schema_path=schema_path,
                location=location,
            )
        )

    def add_warning(message: str, location: Optional[str] = None) -> None:
        messages.append(
            SchemaValidationMessage(
                severity=ValidationSeverity.WARNING,
                message=message,
                schema_path=schema_path,
                location=location,
            )
        )

    run_groups = schema.run_groups or []
    metric_groups = schema.metric_groups or []
    metrics = schema.metrics or []
    perturbations = schema.perturbations or []

    defined_run_group_names: Set[str] = set(schema.name_to_run_group.keys())
    defined_metric_group_names: Set[str] = set(schema.name_to_metric_group.keys())
    defined_metric_names: Set[str] = set(schema.name_to_metric.keys())
    defined_perturbation_names: Set[str] = set(schema.name_to_perturbation.keys())

    # Check for empty names
    for i, run_group in enumerate(run_groups):
        if not run_group.name or not run_group.name.strip():
            add_error(f"Run group at index {i} has empty or whitespace-only name", location=f"run_groups[{i}]")

    for i, metric_group in enumerate(metric_groups):
        if not metric_group.name or not metric_group.name.strip():
            add_error(f"Metric group at index {i} has empty or whitespace-only name", location=f"metric_groups[{i}]")

    for i, metric in enumerate(metrics):
        if not metric.name or not metric.name.strip():
            add_error(f"Metric at index {i} has empty or whitespace-only name", location=f"metrics[{i}]")

    # Check for duplicate names
    seen_run_group_names: Set[str] = set()
    for run_group in run_groups:
        if run_group.name and run_group.name in seen_run_group_names:
            add_error(f"Duplicate run_group name: '{run_group.name}'", location=f"run_groups[{run_group.name}]")
        if run_group.name:
            seen_run_group_names.add(run_group.name)

    seen_metric_group_names: Set[str] = set()
    for metric_group in metric_groups:
        if metric_group.name and metric_group.name in seen_metric_group_names:
            add_error(
                f"Duplicate metric_group name: '{metric_group.name}'", location=f"metric_groups[{metric_group.name}]"
            )
        if metric_group.name:
            seen_metric_group_names.add(metric_group.name)

    seen_metric_names: Set[str] = set()
    for metric in metrics:
        if metric.name and metric.name in seen_metric_names:
            add_error(f"Duplicate metric name: '{metric.name}'", location=f"metrics[{metric.name}]")
        if metric.name:
            seen_metric_names.add(metric.name)

    seen_perturbation_names: Set[str] = set()
    for i, perturbation in enumerate(perturbations):
        if perturbation.name and perturbation.name in seen_perturbation_names:
            add_error(
                f"Duplicate perturbation name: '{perturbation.name}'", location=f"perturbations[{perturbation.name}]"
            )
        if perturbation.name:
            seen_perturbation_names.add(perturbation.name)

    # Validate run_group.subgroups references
    for run_group in run_groups:
        if not run_group.name:
            continue

        subgroups = run_group.subgroups or []
        for i, subgroup_name in enumerate(subgroups):
            if not subgroup_name:
                add_error(
                    f"Empty subgroup reference at index {i}", location=f"run_groups[{run_group.name}].subgroups[{i}]"
                )
            elif subgroup_name not in defined_run_group_names:
                add_error(
                    f"Subgroup '{subgroup_name}' is not defined as a run_group",
                    location=f"run_groups[{run_group.name}].subgroups[{i}]",
                )

    # Validate run_group.metric_groups references
    for run_group in run_groups:
        if not run_group.name:
            continue

        mg_list = run_group.metric_groups or []
        for i, metric_group_name in enumerate(mg_list):
            if not metric_group_name:
                add_error(
                    f"Empty metric_group reference at index {i}",
                    location=f"run_groups[{run_group.name}].metric_groups[{i}]",
                )
            elif metric_group_name not in defined_metric_group_names:
                add_error(
                    f"Metric group '{metric_group_name}' is not defined",
                    location=f"run_groups[{run_group.name}].metric_groups[{i}]",
                )

        hidden_mgs = run_group.subgroup_metric_groups_hidden or []
        for i, hidden_mg_name in enumerate(hidden_mgs):
            if hidden_mg_name and hidden_mg_name not in defined_metric_group_names:
                add_error(
                    f"Hidden metric group '{hidden_mg_name}' is not defined",
                    location=f"run_groups[{run_group.name}].subgroup_metric_groups_hidden[{i}]",
                )

    # Validate metric_group.metrics entries
    for metric_group in metric_groups:
        if not metric_group.name:
            continue

        metrics_list = metric_group.metrics or []
        for i, metric_matcher in enumerate(metrics_list):
            location = f"metric_groups[{metric_group.name}].metrics[{i}]"

            metric_name = metric_matcher.name
            if not metric_name:
                add_error("Empty metric name", location=f"{location}.name")
            elif not is_template_variable(metric_name) and metric_name not in defined_metric_names:
                add_error(
                    f"Metric '{metric_name}' is not defined and is not a template variable", location=f"{location}.name"
                )

            split = metric_matcher.split
            if not split:
                add_error("Empty split value", location=f"{location}.split")
            elif not is_template_variable(split) and split not in VALID_SPLITS:
                add_error(
                    f"Split '{split}' is not valid. Must be one of {sorted(VALID_SPLITS)} "
                    f"or a template variable like ${{main_split}}",
                    location=f"{location}.split",
                )

            pert_name = metric_matcher.perturbation_name
            if pert_name is not None and pert_name:
                if not is_template_variable(pert_name) and pert_name not in defined_perturbation_names:
                    add_error(f"Perturbation '{pert_name}' is not defined", location=f"{location}.perturbation_name")

    # Detect circular references
    cycles = _detect_cycles_in_subgroups(run_groups, schema.name_to_run_group)
    for cycle_start, cycle_path in cycles:
        cycle_str = " -> ".join(cycle_path)
        add_error(
            f"Circular reference detected in subgroups: {cycle_str}", location=f"run_groups[{cycle_start}].subgroups"
        )

    # Optional: Parent/Child partition check
    if check_parent_child_partition:
        for run_group in run_groups:
            if not run_group.name:
                continue

            has_subgroups = len(run_group.subgroups or []) > 0
            has_metric_groups = len(run_group.metric_groups or []) > 0

            if has_subgroups and has_metric_groups:
                add_warning(
                    f"Run group '{run_group.name}' has both subgroups and metric_groups",
                    location=f"run_groups[{run_group.name}]",
                )

    # Optional: Orphan children check
    if check_orphan_children:
        referenced_as_subgroup: Set[str] = set()
        for run_group in run_groups:
            for subgroup_name in run_group.subgroups or []:
                referenced_as_subgroup.add(subgroup_name)

        for run_group in run_groups:
            if not run_group.name:
                continue

            mg_count = len(run_group.metric_groups or [])
            sg_count = len(run_group.subgroups or [])
            is_child = mg_count > 0 and sg_count == 0

            if is_child and run_group.name not in referenced_as_subgroup:
                add_warning(
                    f"Child run_group '{run_group.name}' is not referenced by any parent",
                    location=f"run_groups[{run_group.name}]",
                )

    if strict:
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        if error_messages:
            raise SchemaValidationError(messages)

    return messages


def validate_schema_file(
    schema_path: str,
    *,
    strict: bool = True,
    **kwargs,
) -> List[SchemaValidationMessage]:
    """Convenience function to validate a schema file directly."""
    try:
        schema = read_schema(schema_path)
    except FileNotFoundError:
        msg = SchemaValidationMessage(
            severity=ValidationSeverity.ERROR,
            message=f"Schema file not found: {schema_path}",
            schema_path=schema_path,
        )
        if strict:
            raise SchemaValidationError([msg])
        return [msg]
    except yaml.YAMLError as e:
        msg = SchemaValidationMessage(
            severity=ValidationSeverity.ERROR,
            message=f"Invalid YAML syntax: {e}",
            schema_path=schema_path,
        )
        if strict:
            raise SchemaValidationError([msg])
        return [msg]
    except Exception as e:
        msg = SchemaValidationMessage(
            severity=ValidationSeverity.ERROR,
            message=f"Failed to load schema: {type(e).__name__}: {e}",
            schema_path=schema_path,
        )
        if strict:
            raise SchemaValidationError([msg])
        return [msg]

    return validate_schema(schema, schema_path=schema_path, strict=strict, **kwargs)


def get_all_schema_paths() -> List[str]:
    """Get paths to all schema YAML files included in the HELM package."""
    schema_package = resources.files(SCHEMA_YAML_PACKAGE)
    schema_paths = []

    for item in schema_package.iterdir():
        if item.name.startswith("schema_") and item.name.endswith(".yaml"):
            schema_paths.append(str(schema_package.joinpath(item.name)))

    return sorted(schema_paths)
