import ast
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import dacite
from inspect import cleandoc
import mako.template
import yaml
import importlib_resources as resources

from helm.common.general import hlog
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.augmentations.perturbation_description import PERTURBATION_WORST


# TODO: change to `helm.benchmark.config`
SCHEMA_YAML_PACKAGE: str = "helm.benchmark.static"

# TODO: add heim, vhelm, etc.
SCHEMA_CLASSIC_YAML_FILENAME: str = "schema_classic.yaml"


_ADAPTER_SPEC_PACKAGE = "helm.benchmark.adaptation"
_ADAPTER_SPEC_FILENAME = "adapter_spec.py"
_ADAPTER_SPEC_CLASS_NAME = "AdapterSpec"


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
class TaxonomyInfo:
    # Task (e.g., question answering)
    task: Optional[str] = None

    # Domain - genre (e.g., Wikipedia)
    what: Optional[str] = None

    # Domain - when it was written (e.g., 2010s)
    when: Optional[str] = None

    # Domain - demographics (e.g., web users)
    who: Optional[str] = None

    # Language (e.g., English)
    language: Optional[str] = None


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


@dataclass
class Schema:
    """Specifies information about what to display on the frontend."""

    # Information about each field
    metrics: List[Field]

    # Information about each perturbation
    perturbations: List[Field]

    # Group the metrics
    metric_groups: List[MetricGroup]

    # Group the scenarios
    run_groups: List[RunGroup]

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
    adapter_spec_path = resources.files(_ADAPTER_SPEC_PACKAGE).joinpath(_ADAPTER_SPEC_FILENAME)
    with open(adapter_spec_path, "r") as f:
        contents = f.read()
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
    return resources.files(SCHEMA_YAML_PACKAGE).joinpath(SCHEMA_CLASSIC_YAML_FILENAME)


def read_schema(schema_path: str) -> Schema:
    # TODO: merge in model metadata from `model_metadata.yaml`
    hlog(f"Reading schema file {schema_path}...")
    with open(schema_path, "r") as f:
        raw = yaml.safe_load(f)
    schema = dacite.from_dict(Schema, raw)
    if schema.adapter:
        hlog(f"WARNING: The `adapter` field is deprecated and should be removed from schema file {schema_path}")
    return dataclasses.replace(schema, adapter=get_adapter_fields())
