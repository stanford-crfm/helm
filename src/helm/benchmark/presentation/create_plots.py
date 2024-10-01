# mypy: check_untyped_defs = False
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import json
import os
from typing import List, Dict, Optional, Any, Callable, Union, Mapping, Tuple, Set

import numpy as np
from scipy.stats import pearsonr

from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.model_metadata_registry import MODEL_NAME_TO_MODEL_METADATA

try:
    import colorcet
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["plots"])


sns.set_style("whitegrid")

DOWN_ARROW = "\u2193"
UP_ARROW = "\u2191"
metric_group_to_label = {
    "Accuracy": f"Accuracy {UP_ARROW}",
    "Calibration": f"Calibration error {DOWN_ARROW}",
    "Robustness": f"Robustness {UP_ARROW}",
    "Fairness": f"Fairness {UP_ARROW}",
    "Bias": f"Bias (gender repr.) {DOWN_ARROW}",
    "Toxicity": f"Toxicity {DOWN_ARROW}",
    "Efficiency": f"Inference time (s) {DOWN_ARROW}",
}
all_metric_groups = list(metric_group_to_label.keys())
AGGREGATE_WIN_RATE_COLUMN = 1


@dataclass
class Column:
    """Values and metadata for each column of the table."""

    name: str
    group: str
    metric: str
    values: np.ndarray
    lower_is_better: Optional[bool]


@dataclass
class Table:
    """Column-based representation of a standard run-group table. See summarize.py for exact documentation."""

    adapters: List[str]
    columns: List[Column]
    mean_win_rates: Optional[np.ndarray] = None


def parse_table(raw_table: Dict[str, Any]) -> Table:
    """Convert raw table dict to a Table. Ignores strongly contaminated table entries."""

    def get_cell_values(cells: List[dict]) -> List[Any]:
        values = []
        for cell in cells:
            value = cell["value"] if "value" in cell else np.nan
            if "contamination_level" in cell and cell["contamination_level"] == "strong":
                value = np.nan
            values.append(value)
        return values

    adapters: Optional[List[str]] = None
    columns: List[Column] = []
    mean_win_rates: Optional[np.ndarray] = None
    for column_index, (header_cell, *column_cells) in enumerate(zip(raw_table["header"], *raw_table["rows"])):
        cell_values = get_cell_values(column_cells)
        if column_index == 0:
            adapters = cell_values
        elif column_index == AGGREGATE_WIN_RATE_COLUMN and "win rate" in header_cell["value"]:
            mean_win_rates = np.array(cell_values)
        else:
            assert "metadata" in header_cell
            name = header_cell["value"]
            group = header_cell["metadata"]["run_group"]
            metric = header_cell["metadata"]["metric"]
            lower_is_better = header_cell["lower_is_better"] if "lower_is_better" in header_cell else None
            columns.append(Column(name, group, metric, np.array(cell_values), lower_is_better))
    assert adapters is not None

    return Table(adapters, columns, mean_win_rates)


def get_color_palette(n_colors: int) -> sns.palettes._ColorPalette:
    if n_colors < 6:
        return sns.color_palette("colorblind", n_colors=n_colors)
    else:
        return sns.color_palette(colorcet.glasbey_warm, n_colors=n_colors)


def draw_box_plot(
    x_to_ys: Mapping[str, Union[List[float], np.ndarray]], ax: matplotlib.axis.Axis, rotate_xticklabels: bool = True
):
    """Given a mapping from string to floats, draw a box plot on the given axis ax. For instance, this might be a
    mapping from scenario_name to a list of model accuracies in which case the box plot captures aggregate model
    performance and highlights outliers."""
    xs: List[str] = []
    all_ys: List[List[float]] = []
    for x, ys in x_to_ys.items():
        ys = [y for y in ys if not np.isnan(y)]
        if ys:
            xs.append(x)
            all_ys.append(ys)
            ax.scatter([len(xs)] * len(ys), ys, c="#bbb")
    ax.boxplot(all_ys, boxprops={"linewidth": 2}, medianprops={"linewidth": 2})
    if rotate_xticklabels:
        ax.set_xticklabels(xs, rotation=-30, ha="left")
    else:
        ax.set_xticklabels(xs)


class Plotter:
    """
    Main class producing plots. Each create_*() method reads group data from base_path and creates a plot at the
    save_path. create_all_plots() runs all these functions at once.
    """

    def __init__(self, base_path: str, save_path: str, plot_format: str):
        self.base_path = base_path
        self.save_path = save_path
        self.plot_format = plot_format
        self._tables_cache: Dict[str, Dict[str, Table]] = {}

    def get_group_tables(self, group_name: str) -> Dict[str, Table]:
        """Reads and parses group tables. Uses _tables_cache to avoid reprocessing the same table multiple times."""
        if group_name in self._tables_cache:
            return self._tables_cache[group_name]
        with open(os.path.join(self.base_path, "groups", f"{group_name}.json")) as fp:
            tables = json.load(fp)

        name_to_table: Dict[str, Table] = {}
        for table in tables:
            name_to_table[table["title"]] = parse_table(table)

        return name_to_table

    def save_figure(self, fig: plt.Figure, name: str):
        """Save and close a figure."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        fig.savefig(os.path.join(self.save_path, f"{name}.{self.plot_format}"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    def create_accuracy_v_x_plots(self):
        """
        For each metric group, create a scatter plot with Accuracy on the x-axis and that metric group on the y-axis.
        Each point corresponds to a model-scenario pair, colored by the scenario.
        """
        tables = self.get_group_tables("core_scenarios")
        metric_groups_shown = [metric_group for metric_group in all_metric_groups if metric_group != "Accuracy"]

        num_columns = 3
        num_rows = (len(metric_groups_shown) - 1) // num_columns + 1
        fig, axarr = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 3.5 * num_rows))
        all_groups = [column.group for column in tables["Accuracy"].columns]
        palette = get_color_palette(len(all_groups))
        for i, metric_group in enumerate(metric_groups_shown):
            table: Table = tables[metric_group]
            ax = axarr[i // num_columns][i % num_columns]
            for column in table.columns:
                if metric_group == "Bias" and column.metric != "Representation (gender)":  # only show gender bias
                    continue
                accuracy_column: Column = [c for c in tables["Accuracy"].columns if c.group == column.group][0]
                group_idx = all_groups.index(column.group)
                ax.scatter(
                    accuracy_column.values, column.values, color=palette[group_idx], alpha=0.8, label=column.group
                )

            if metric_group in ["Robustness", "Fairness"]:
                ax.plot([0, 1], [0, 1], ls="--", c="gray", zorder=-1)
            if metric_group == "Bias":
                ax.axhline(0.5, ls="--", c="gray", zorder=-1)

            ax.set_xlabel("Accuracy", fontsize=14)
            ax.set_ylabel(metric_group_to_label[metric_group], fontsize=14)
            ax.set_xlim(-0.1, 1.1)

        # create dummy lines to display a single legend for all plots
        lines = [
            matplotlib.lines.Line2D([], [], color="white", marker="o", markersize=10, markerfacecolor=color)
            for color in palette
        ]
        axarr[0][0].legend(
            lines, all_groups, title="Scenarios", loc="lower left", bbox_to_anchor=(0, 1), ncol=6, numpoints=1
        )

        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self.save_figure(fig, "accuracy_v_x")

    def create_correlation_plots(self):
        """
        For each metric group, create a box-plot aggregating how correlated that metric group is with each other
        metric_group. Individual point correspond to the correlation (across models) of the two metric groups on a
        single scenario.
        """
        tables = self.get_group_tables("core_scenarios")
        metric_groups_shown = all_metric_groups

        num_columns = 3
        num_rows = (len(metric_groups_shown) - 2) // num_columns + 1
        fig, axarr = plt.subplots(num_rows, num_columns, figsize=(5.5 * num_columns, 4.5 * num_rows))

        for i, metric_group_1 in enumerate(metric_groups_shown[:-1]):
            ax = axarr[i // num_columns][i % num_columns]
            group_to_values_1: Dict[str, np.ndarray] = {}
            for column in tables[metric_group_1].columns:
                if metric_group_1 == "Bias" and column.metric != "Representation (gender)":
                    continue
                group_to_values_1[column.group] = column.values

            metric_group_to_correlations: Dict[str, np.ndarray] = defaultdict(list)
            for j, metric_group_2 in enumerate(metric_groups_shown):
                for column in tables[metric_group_2].columns:
                    if metric_group_2 == "Bias" and column.metric != "Representation (gender)":
                        continue
                    if column.group not in group_to_values_1:
                        continue
                    values_1 = group_to_values_1[column.group]
                    values_2 = column.values
                    valid_values = np.logical_and(~np.isnan(values_1), ~np.isnan(values_2))
                    if sum(valid_values) >= 2:
                        correlation = pearsonr(values_1[valid_values], values_2[valid_values])[0]
                        label = metric_group_to_label[metric_group_2]
                        metric_group_to_correlations[label].append(correlation)
            draw_box_plot(metric_group_to_correlations, ax)
            ax.set_title(metric_group_to_label[metric_group_1])
            if i % num_columns == 0:
                ax.set_ylabel("Pearson correlation")

        fig.subplots_adjust(wspace=0.25, hspace=0.45)
        self.save_figure(fig, "metric_correlation")

    def create_leaderboard_plots(self):
        """Display the model mean win rates for each group as a bar chart."""
        tables = self.get_group_tables("core_scenarios")

        metric_groups_shown = [metric_group for metric_group in all_metric_groups if metric_group != "Efficiency"]
        num_columns = 3
        num_rows = (len(metric_groups_shown) - 1) // num_columns + 1
        fig, axarr = plt.subplots(num_rows, num_columns, figsize=(4 * num_columns, 6.7 * num_rows))
        for i, metric_group in enumerate(metric_groups_shown):
            win_rates, models = [], []
            for win_rate, model in sorted(zip(tables[metric_group].mean_win_rates, tables[metric_group].adapters)):
                if not np.isnan(win_rate):
                    win_rates.append(win_rate)
                    models.append(model)
            ax = axarr[i // num_columns][i % num_columns]
            ax.plot([0, 1], [0, len(models) - 1], ls="--", c="#bbb", zorder=0)
            ax.barh(models, win_rates, label=models)
            ax.set_xlim(-0.1, 1.1)
            ax.set_title(metric_group)
        fig.subplots_adjust(wspace=1.8, hspace=0.15)
        self.save_figure(fig, "model_ranking_all")

    def create_accuracy_v_model_property_plot(
        self,
        property_name: str,
        model_name_to_property: Callable[[str], Any],
        cumulative: bool = False,
        logscale: bool = False,
        annotate_models: bool = True,
    ):
        """
        Plot the accuracy of each scenario over some model property (e.g., number of parameters).
        Args:
            property_name: Property name displayed as x-label.
            model_name_to_property: A function that maps the name of the model to the property we use for the plot.
            cumulative: Plot the best accuracy achieved by a model with at most that property values (useful for dates).
            logscale: Whether we use a logscale for the x-axis.
            annotate_models: For each unique property value, add a text annotation with the corresponding model names.
        """
        fig, ax = plt.subplots(1, 1, figsize=(11, 4))
        milestones: Dict[Any, Set[str]] = defaultdict(set)  # keep track of the models with each property value
        table = self.get_group_tables("core_scenarios")["Accuracy"]
        palette = get_color_palette(len(table.columns))
        for column, color in zip(table.columns, palette):
            data: List[Tuple[Any, float]] = []
            for model_name, accuracy in zip(table.adapters, column.values):
                key = model_name_to_property(model_name)
                if key is None or np.isnan(accuracy):
                    continue
                data.append((key, accuracy))
                milestones[key].add(model_name)
            data.sort()
            xs: List[Any] = []
            ys: List[float] = []
            if cumulative:
                for now in list(dict.fromkeys(key for key, _ in data)):
                    xs.append(now)
                    ys.append(max(y for (x, y) in data if x <= now))
            else:
                for x, y in data:
                    xs.append(x)
                    ys.append(y)
            plot_func = ax.semilogx if logscale else ax.plot
            plot_func(xs, ys, label=column.group, color=color, marker="o")

        for key, model_names in sorted(milestones.items()):
            ax.axvline(x=key, ls="--", c="#bbb", zorder=0)
            if annotate_models:
                ax.text(key, 1.01, "/".join(model_names), rotation=40)

        # sort the legend according to the left-most value of each plot (makes it easier to visually map names to lines)
        handles, labels = ax.get_legend_handles_labels()
        legend_order = np.argsort([-h.get_data()[1][-1] for h in handles])
        ax.legend(
            [handles[i] for i in legend_order],
            [labels[i] for i in legend_order],
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )
        property_save_name = property_name.replace(" ", "_").lower()
        ax.set_xlabel(property_name)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        self.save_figure(fig, f"accuracy_over_{property_save_name}")

    def create_all_accuracy_v_model_property_plots(self):
        """
        Create accuracy-vs-property plots for: release date, #parameters, thePile perplexity.
        In all cases, we use a coarse value for the property to make the plot text annotations cleaner.
        """

        def get_model_release_date(model_name: str) -> Optional[date]:
            """Maps a model name to the month of model release."""
            release_date = MODEL_NAME_TO_MODEL_METADATA[model_name].release_date
            if release_date is None:
                return None
            return release_date.replace(day=1)

        def get_model_size(model_name: str) -> Optional[int]:
            """Maps a model name to the number of parameters, rounding to the nearest leading digit."""
            size = MODEL_NAME_TO_MODEL_METADATA[model_name].num_parameters
            if size is None:
                return None
            grain = 10 ** (len(str(size)) - 1)
            return round(size / grain) * grain  # only look at first digit

        # Read the perplexity of The Pile according to each model
        bpb_table = self.get_group_tables("the_pile")["The Pile"]
        model_to_bpb: Dict[str, float] = {
            model: bpb for model, bpb in zip(bpb_table.adapters, bpb_table.columns[0].values)
        }

        def get_model_perplexity(model_name: str) -> Optional[float]:
            """Maps a model name to the perplexity of The Pile of parameters, rounding based on some granularity."""
            if model_name not in model_to_bpb or np.isnan(model_to_bpb[model_name]):
                return None
            bpb = model_to_bpb[model_name]
            grain = 0.016
            return round(bpb / grain) * grain

        annotate_models = True if self.plot_format == "pdf" else False
        self.create_accuracy_v_model_property_plot(
            "Release date",
            get_model_release_date,
            cumulative=True,
            annotate_models=annotate_models,
        )
        self.create_accuracy_v_model_property_plot(
            "Num parameters",
            get_model_size,
            cumulative=True,
            logscale=True,
            annotate_models=annotate_models,
        )
        self.create_accuracy_v_model_property_plot(
            "The Pile perplexity",
            get_model_perplexity,
            logscale=True,
            annotate_models=annotate_models,
        )

    def create_accuracy_v_access_bar_plot(self):
        """
        For each scenario, plot the best model performance for each access level (e.g., closed). We plot both the
        performance of the best model chosen for a particular scenario (transparent, in the back) as well as the best
        overall model at that access level.
        """
        table = self.get_group_tables("core_scenarios")["Accuracy"]

        all_groups = [column.group for column in table.columns]
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        palette = get_color_palette(n_colors=3)
        access_levels = ["open", "limited", "closed"]

        for i, access_level in enumerate(access_levels):
            model_indices: List[int] = [
                idx
                for idx, model in enumerate(table.adapters)
                if MODEL_NAME_TO_MODEL_METADATA[model].access == access_level
            ]
            best_model_index = model_indices[table.mean_win_rates[model_indices].argmax()]

            xs = np.arange(len(all_groups))
            width = 0.25
            ys, ys_single = [], []
            for column in table.columns:
                ys.append(column.values[model_indices].max())
                ys_single.append(column.values[best_model_index])
            ax.bar(xs + (i - 1) * width, ys, width, color=palette[i], alpha=0.5)
            ax.bar(xs + (i - 1) * width, ys_single, width, label=access_level, color=palette[i])

        ax.set_ylabel("Accuracy")
        ax.set_xticks(xs, all_groups, rotation=-20, ha="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.61, 0.99))
        self.save_figure(fig, "accuracy_v_access")

    def create_task_summary_plots(self):
        """For each metric group, create a box plot with scenario performance across models."""
        tables = self.get_group_tables("core_scenarios")
        metric_groups = ["Accuracy", "Calibration", "Robustness", "Fairness", "Bias", "Toxicity"]
        num_columns = 2
        num_rows = (len(metric_groups) - 1) // num_columns + 1
        fig, axarr = plt.subplots(num_rows, num_columns, figsize=(7 * num_columns, 4.5 * num_rows))
        for i, metric_group in enumerate(metric_groups):
            ax = axarr[i // num_columns][i % num_columns]
            table = tables[metric_group]
            group_to_accuracies: Dict[str, np.ndarray] = {
                column.group: column.values
                for column in table.columns
                if not (metric_group == "Bias" and column.metric != "Representation (gender)")
            }
            draw_box_plot(group_to_accuracies, ax)
            ax.set_title(metric_group_to_label[metric_group])

        fig.subplots_adjust(hspace=0.7)
        self.save_figure(fig, "generic_summary")

    def create_targeted_eval_plots(self):
        """Create a box plots with scenario accuracy across models for a range of targeted evaluations."""
        fig, axd = plt.subplot_mosaic([["language", "knowledge"], ["reasoning", "reasoning"]], figsize=(12, 7))
        for targeted_eval in ["language", "knowledge", "reasoning"]:
            table = self.get_group_tables(targeted_eval)["Accuracy"]
            ax = axd[targeted_eval]

            group_to_accuracies: Dict[str, np.ndarray] = {}
            for column in table.columns:
                arrow = DOWN_ARROW if column.lower_is_better else UP_ARROW
                group = f"{column.group}\n({column.metric} {arrow})"
                group_to_accuracies[group] = column.values

            draw_box_plot(group_to_accuracies, ax)
            ax.set_title(targeted_eval.capitalize())
            ax.set_ylim(-0.1, 3.6 if targeted_eval == "language" else 1.1)

        fig.subplots_adjust(hspace=0.75)
        self.save_figure(fig, "targeted_evals")

    def create_copyright_plot(self):
        """Plot copyright metrics across models."""
        table = self.get_group_tables("harms")["Copyright metrics"]
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        group_to_values: Dict[str, np.ndarray] = {}
        for column in table.columns:
            if "dist" in column.metric:
                continue
            arrow = DOWN_ARROW if column.lower_is_better else UP_ARROW
            group = f"{column.group}\n({column.metric} {arrow})"
            group_to_values[group] = column.values
        draw_box_plot(group_to_values, ax, rotate_xticklabels=False)
        ax.set_title("Copyright")
        self.save_figure(fig, "copyright")

    def create_bbq_plot(self):
        """Plot BBQ metrics across models."""
        table = self.get_group_tables("harms")["BBQ metrics"]
        n = len(table.columns)
        fig, axarr = plt.subplots(1, n, figsize=(3.5 * n, 7))
        for i, column in enumerate(table.columns):
            ax = axarr[i]
            indices = np.argsort(column.values)
            indices = indices[: -np.isnan(column.values).sum()]  # remove nans from the end
            models = np.array(table.adapters)[indices]
            values = column.values[indices]
            ax.barh(models, values)
            ax.set_title(f"{column.metric} {DOWN_ARROW}")
        fig.subplots_adjust(wspace=1.85)
        self.save_figure(fig, "bbq_bars")

    def create_in_context_examples_plot(self):
        """
        Plot model performance as a function of in-context examples used. One plot per scenario, one line per model.
        We retrieve the actual average number of in-context examples used from the "General information" table.
        """
        tables = self.get_group_tables("ablation_in_context")

        group_to_num_examples: Dict[str, np.ndarray] = {}
        for column in tables["General information"].columns:
            if "# train" in column.name:
                group_to_num_examples[column.group] = column.values

        table = tables["Accuracy"]
        n = len(table.columns)
        fig, axarr = plt.subplots(1, n, figsize=(3.5 * n, 2.5))
        for i, column in enumerate(table.columns):
            model_examples_to_accuracy: Dict[str, Dict[float, float]] = defaultdict(dict)
            for adapter, accuracy, num_examples in zip(
                table.adapters, column.values, group_to_num_examples[column.group]
            ):
                model = adapter.split(" [")[0]
                model_examples_to_accuracy[model][num_examples] = accuracy

            ax = axarr[i]
            for model, examples_to_accuracy in model_examples_to_accuracy.items():
                if "UL2" in model:
                    continue
                offset = 2
                xs: List[int] = []
                ys: List[float] = []
                for x, y in sorted(examples_to_accuracy.items()):
                    xs.append(x)
                    ys.append(y)
                ax.semilogx([x + offset for x in xs], ys, label=model, marker="o", base=2)
                xs_max = [0] + [2**i for i in range(5)]
                ax.set_xticks([x + offset for x in xs_max], xs_max)
                ax.set_title(column.group)
                ax.set_ylabel(column.metric)
                ax.set_xlabel("#in-context examples")
                if i == 0:
                    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.45))
        fig.subplots_adjust(wspace=0.32, hspace=0.5)
        self.save_figure(fig, "in_context_ablations")

    def create_mc_ablations_plot(self):
        """For each scenario, plot model performance (as a bar plot) for each multiple-choice adaptation method."""
        table = self.get_group_tables("ablation_multiple_choice")["Accuracy"]

        num_columns = 4
        num_rows = (len(table.columns) - 1) // num_columns + 1
        fig, axarr = plt.subplots(num_rows, num_columns, figsize=(4 * num_columns, 3 * num_rows))

        method_to_label = {
            "multiple_choice_joint": "Multiple Choice Joint",
            "multiple_choice_separate_original": "Multiple Choice Separate",
            "multiple_choice_separate_calibrated": "Multiple Choice Separate Calibrated",
        }
        palette = get_color_palette(len(method_to_label))
        width = 0.2
        for i, column in enumerate(table.columns):
            ax = axarr[i // num_columns][i % num_columns]
            for j, method in enumerate(method_to_label):
                models: List[str] = []
                ys: List[float] = []
                for adapter, accuracy in zip(table.adapters, column.values):
                    if method not in adapter:
                        continue
                    models.append(adapter.split(" [")[0])
                    ys.append(accuracy)
                xs = np.arange(len(models))
                ax.bar(xs + (j - 1) * width, ys, width, color=palette[j], label=method_to_label[method])
                ax.set_xticks(xs, models, rotation=-20, ha="left")
                ax.set_title(f"{column.group} ({column.metric})")
                if i == 0:
                    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.4))
        fig.subplots_adjust(wspace=0.25, hspace=0.65)
        self.save_figure(fig, "mc_ablations")

    def create_constrast_set_plots(self):
        """For each contrast set scenario, plot the accuracy and robustness of each model on a scatter plot."""
        tables = self.get_group_tables("robustness_contrast_sets")
        fig, axarr = plt.subplots(1, 2, figsize=(7, 3))
        for ax, (table_name, table) in zip(axarr, tables.items()):
            xs = [column for column in table.columns if column.name == "EM"][0].values
            ys = [column for column in table.columns if column.name == "EM (Robustness)"][0].values
            ax.scatter(xs, ys)
            ax.plot([0, 1], [0, 1], color="gray", ls="--")
            ax.set_title(table.columns[0].group)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Robustness")
        fig.subplots_adjust(wspace=0.25)
        self.save_figure(fig, "contrast_sets")

    def create_all_plots(self):
        """Create all the plots used in the HELM paper."""
        self.create_accuracy_v_x_plots()
        self.create_correlation_plots()
        self.create_leaderboard_plots()
        self.create_all_accuracy_v_model_property_plots()
        self.create_accuracy_v_access_bar_plot()
        self.create_task_summary_plots()
        self.create_targeted_eval_plots()
        self.create_copyright_plot()
        self.create_bbq_plot()
        self.create_in_context_examples_plot()
        self.create_mc_ablations_plot()
        self.create_constrast_set_plots()


def main():
    """
    This script creates the plots used in the HELM paper (https://arxiv.org/abs/2211.09110).
    It should be run _after_ running `summarize.py` with the same `benchmark_output` and `suite` arguments and through
    the top-level command `helm-create-plots`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, help="Path to benchmarking output", default="benchmark_output")
    parser.add_argument("--suite", type=str, help="Name of the suite that we are plotting", required=True)
    parser.add_argument("--plot-format", help="Format for saving plots", default="png", choices=["png", "pdf"])
    args = parser.parse_args()
    register_builtin_configs_from_helm_package()
    base_path = os.path.join(args.output_path, "runs", args.suite)
    if not os.path.exists(os.path.join(base_path, "groups")):
        hlog(f"ERROR: Could not find `groups` directory under {base_path}. Did you run `summarize.py` first?")
        return
    save_path = os.path.join(base_path, "plots")
    plotter = Plotter(base_path=base_path, save_path=save_path, plot_format=args.plot_format)
    plotter.create_all_plots()


if __name__ == "__main__":
    main()
