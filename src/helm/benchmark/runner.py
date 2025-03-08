import dacite
import json
import math
import os
import traceback
import typing
from collections import Counter
import dataclasses
from typing import Any, Dict, List
from queue import PriorityQueue
import numpy as np
import torch

from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from helm.benchmark.scenarios.scenario import (
    EVAL_SPLITS,
    TRAIN_SPLIT,
    Scenario,
    create_scenario,
    Instance,
    get_scenario_cache_path,
    with_instance_ids,
)
from helm.benchmark.adaptation.adapters.adapter import Adapter
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.executor import ExecutionSpec, Executor
from helm.benchmark.annotation_executor import AnnotationExecutionSpec, AnnotationExecutor
from helm.benchmark.metrics.dry_run_metrics import DryRunMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, create_metric, Stat
from helm.benchmark.window_services.tokenizer_service import TokenizerService


LATEST_SYMLINK: str = "latest"
_BENCHMARK_OUTPUT_PATH: str = "benchmark_output"
_CACHED_MODELS_FOLDER: str = "models"


def get_benchmark_output_path() -> str:
    """Get the benchmark output path.

    Many run spec functions need to know the benchmark output path,
    but there is no way to pass it via  the run spec function,
    so instead the run spec function should read this global variable."""
    return _BENCHMARK_OUTPUT_PATH


def get_cached_models_path() -> str:
    """Get the cached models pat within the benchmark output path."""
    path: str = os.path.join(get_benchmark_output_path(), _CACHED_MODELS_FOLDER)
    ensure_directory_exists(path)
    return path


def set_benchmark_output_path(benchmark_output_path: str) -> None:
    """Set the benchmark output path."""
    global _BENCHMARK_OUTPUT_PATH
    _BENCHMARK_OUTPUT_PATH = benchmark_output_path


class RunnerError(Exception):
    """Error that happens in the Runner."""

    pass


def remove_stats_nans(stats: List[Stat]) -> List[Stat]:
    """Return a new list of stats with stats with NaNs removed.

    Python's stdlib json.dumps() will produce invalid JSON when serializing a NaN. See:

    - https://github.com/stanford-crfm/helm/issues/1765
    - https://bugs.python.org/issue40633
    - https://docs.python.org/3/library/json.html#infinite-and-nan-number-values"""
    result: List[Stat] = []
    for stat in stats:
        if math.isnan(stat.sum):
            hlog(f"WARNING: Removing stat {stat.name.name} because its value is NaN")
            continue
        result.append(stat)
    return result


def remove_per_instance_stats_nans(per_instance_stats_list: List[PerInstanceStats]) -> List[PerInstanceStats]:
    """Return a new list of PerInstanceStats with stats with NaNs removed.

    Python's stdlib json.dumps() will produce invalid JSON when serializing a NaN. See:

    - https://github.com/stanford-crfm/helm/issues/1765
    - https://bugs.python.org/issue40633
    - https://docs.python.org/3/library/json.html#infinite-and-nan-number-values"""
    result: List[PerInstanceStats] = []
    for per_instance_stats in per_instance_stats_list:
        result.append(dataclasses.replace(per_instance_stats, stats=remove_stats_nans(per_instance_stats.stats)))
    return result


def downsample_eval_instances(
    instances: List[Instance], max_eval_instances: int, eval_splits: List[str]
) -> List[Instance]:
    """
    Get the instances necessary for this run:
    Train instances (split=train): keep all (if any) for in-context learning
    Eval instances (split=valid or test): keep at most `max_eval_instances` specified in `AdapterSpec` by sampling
    Return the resulting train and eval instances.
    """
    all_train_instances: List[Instance] = [instance for instance in instances if instance.split == TRAIN_SPLIT]

    all_eval_instances: List[Instance] = [instance for instance in instances if instance.split in eval_splits]
    if len(all_eval_instances) > max_eval_instances:
        # The random sampling includes instances monotonically.
        np.random.seed(0)
        selected_eval_instances = list(
            np.random.choice(
                all_eval_instances,  # type: ignore
                max_eval_instances,
                replace=False,
            )
        )
    else:
        selected_eval_instances = all_eval_instances

    hlog(
        f"{len(instances)} instances, "
        f"{len(all_train_instances)} train instances, "
        f"{len(selected_eval_instances)}/{len(all_eval_instances)} eval instances"
    )

    return all_train_instances + selected_eval_instances

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    request_state: Any=dataclasses.field(compare=False)

class Runner:
    """
    The main entry point for running the entire benchmark.  Mostly just
    dispatches to other classes.
    """

    def __init__(
        self,
        execution_spec: ExecutionSpec,
        output_path: str,
        suite: str,
        skip_instances: bool,
        cache_instances: bool,
        cache_instances_only: bool,
        skip_completed_runs: bool,
        exit_on_error: bool,
    ):
        self.executor = Executor(execution_spec)
        self.annotator_executor = AnnotationExecutor(
            AnnotationExecutionSpec(
                local_path=execution_spec.local_path if execution_spec.local_path is not None else "",
                parallelism=execution_spec.parallelism,
                dry_run=execution_spec.dry_run,
                sqlite_cache_backend_config=execution_spec.sqlite_cache_backend_config,
                mongo_cache_backend_config=execution_spec.mongo_cache_backend_config,
            )
        )
        self.dry_run: bool = execution_spec.dry_run
        self.tokenizer_service = TokenizerService(self.executor.service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.service, execution_spec.auth)
        self.skip_instances: bool = skip_instances
        self.cache_instances: bool = cache_instances
        self.cache_instances_only: bool = cache_instances_only
        self.skip_completed_runs: bool = skip_completed_runs
        self.exit_on_error: bool = exit_on_error

        ensure_directory_exists(output_path)
        self.output_path = output_path

        # Decide where to save input instances
        self.instances_path: str = os.path.join(output_path, "scenario_instances")
        ensure_directory_exists(self.instances_path)

        # Output the results under a folder with the name of the suite
        self.runs_path: str = os.path.join(output_path, "runs", suite)

        # The path where to cache files needs to compute metrics, e.g., human evaluation results
        self.eval_cache_path: str = os.path.join(self.runs_path, "eval_cache")
        ensure_directory_exists(self.eval_cache_path)

    def _get_run_path(self, run_spec: RunSpec) -> str:
        return os.path.join(self.runs_path, run_spec.name)

    def _is_run_completed(self, run_path: str):
        """Return whether the run was previously completed.

        A run is completed if all of the expected output files exist."""
        if not os.path.isdir(run_path):
            return False
        output_paths = [
            os.path.join(run_path, "run_spec.json"),
            os.path.join(run_path, "scenario.json"),
            os.path.join(run_path, "scenario_state.json"),
            os.path.join(run_path, "stats.json"),
            os.path.join(run_path, "per_instance_stats.json"),
        ]
        for output_path in output_paths:
            if not os.path.exists(output_path):
                return False
        return True

    def run_all(self, run_specs: List[RunSpec]):
        failed_run_specs: List[RunSpec] = []

        for run_spec in tqdm(run_specs, disable=None):
            try:
                with htrack_block(f"Running {run_spec.name}"):
                    self.run_one(run_spec)
            except Exception as e:
                if self.exit_on_error:
                    raise e
                else:
                    hlog(f"Error when running {run_spec.name}:\n{traceback.format_exc()}")
                    failed_run_specs.append(run_spec)
        if not self.exit_on_error and failed_run_specs:
            failed_runs_str = ", ".join([f'"{run_spec.name}"' for run_spec in failed_run_specs])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")

    def run_one(self, run_spec: RunSpec):
        run_path: str = self._get_run_path(run_spec)
        if self.skip_completed_runs and self._is_run_completed(run_path):
            hlog(f"Skipping run {run_spec.name} because run is completed and all output files exist.")
            return
        ensure_directory_exists(run_path)
        
        # Check adaptive mode
        if run_spec.adaptive_mode and self.executor.execution_spec.parallelism > 1:
            hlog("Adaptive mode is not supported with parallelism > 1. Running with parallelism=1.")

        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario_spec)

        # This 'output_path' will be used when the model's input instances are saved.
        args_str = ",".join([f"{k}={v}" for k, v in sorted(run_spec.scenario_spec.args.items())])
        scenario_name_with_args = f"{scenario.name}:{args_str}" if args_str else f"{scenario.name}"
        input_instances_output_path = os.path.join(self.instances_path, scenario_name_with_args)
        input_instances_file_path = os.path.join(input_instances_output_path, "input_instances.json")

        instances: List[Instance]
        if self.skip_instances:
            instances = []
        else:
            if self.cache_instances and os.path.exists(input_instances_file_path):
                with open(input_instances_file_path) as f:
                    json_instances: List[Dict[str, Any]] = json.load(f)
                instances = [dacite.from_dict(Instance, instance) for instance in json_instances]
            else:
                # Create the instances of the scenario
                scenario_output_path = get_scenario_cache_path(self.output_path, scenario.name)
                with htrack_block("scenario.get_instances"):
                    instances = scenario.get_instances(scenario_output_path)
        if self.cache_instances and not os.path.exists(input_instances_file_path):
            # Save instances to file
            ensure_directory_exists(input_instances_output_path)
            write(
                os.path.join(input_instances_file_path),
                json.dumps([asdict_without_nones(instance) for instance in instances], indent=2),
            )
        if self.cache_instances_only:
            return  # Exit after saving the instances.

        # Give each instance a unique ID
        if any([instance.id is None for instance in instances]):
            instances = with_instance_ids(instances)

        # Get the instances necessary for this run.
        max_eval_instances = run_spec.adapter_spec.max_eval_instances
        eval_splits = run_spec.adapter_spec.eval_splits or EVAL_SPLITS
        if max_eval_instances is not None:
            instances = downsample_eval_instances(instances, max_eval_instances, eval_splits)

        # Data preprocessing
        instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
            instances, self.executor.execution_spec.parallelism
        )

        # Adapt (convert to requests)
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)
        request_states: List[RequestState] = adapter.adapt(instances, self.executor.execution_spec.parallelism)

        if run_spec.adaptive_mode:
            scenario_state, stats, per_instance_stats, adaptive_trajectory = self.run_execution_adaptive(run_spec, request_states)
        else:
            scenario_state, stats, per_instance_stats = self.run_execution(run_spec, request_states)

        # Check that there aren't duplicate `Stat`s
        # Note: doesn't catch near misses.
        metric_counts: typing.Counter[MetricName] = Counter([stat.name for stat in stats])
        for metric_name, count in metric_counts.items():
            if count > 1:
                hlog(f"WARNING: duplicate metric name {metric_name}")

        # Print out the number of stats
        hlog(f"Generated {len(stats)} stats.")

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

        # Write out scenario
        write(os.path.join(run_path, "scenario.json"), json.dumps(asdict_without_nones(scenario), indent=2))

        # Write scenario state
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))

        write(
            os.path.join(run_path, "stats.json"),
            json.dumps([asdict_without_nones(stat) for stat in remove_stats_nans(stats)], indent=2),
        )
        write(
            os.path.join(run_path, "per_instance_stats.json"),
            json.dumps(list(map(asdict_without_nones, remove_per_instance_stats_nans(per_instance_stats))), indent=2),
        )
        
        if run_spec.adaptive_mode:
            write(
                os.path.join(run_path, "adaptive_trajectory.json"),
                json.dumps(adaptive_trajectory, indent=2),
            )

        cache_stats.print_status()

    def run_execution(
        self, 
        run_spec: RunSpec,
        request_states: List[RequestState],
    ):
        scenario_state: ScenarioState = ScenarioState(
            adapter_spec=run_spec.adapter_spec,
            request_states=request_states,
            annotator_specs=run_spec.annotators,
        )

        # Execute (fill up results)
        scenario_state = self.executor.execute(scenario_state)

        # Annotate (post-process the results)
        scenario_state = self.annotator_executor.execute(scenario_state)

        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[MetricInterface] = (
            [DryRunMetric()] if self.dry_run else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
        )
        stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []
        with htrack_block(f"{len(metrics)} metrics"):
            for metric in metrics:
                with htrack_block(metric):
                    metric_result: MetricResult = metric.evaluate(
                        scenario_state,
                        self.metric_service,
                        self.eval_cache_path,
                        self.executor.execution_spec.parallelism,
                    )
                    stats.extend(metric_result.aggregated_stats)
                    per_instance_stats.extend(metric_result.per_instance_stats)

        return scenario_state, stats, per_instance_stats

    def _compute_negative_fisher_information(
        self,
        model_ability: float,
        instance_difficulty: float,
    ):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        llh = sigmoid(model_ability + instance_difficulty)
        return -llh * (1 - llh)

    def _construct_request_state_queue(
        self,
        request_states: List[RequestState],
        model_ability: float,
        priority: str = "adaptive",
    ):
        if priority == "random":
            priorities = np.random.rand(len(request_states))

        elif priority == "default":
            priorities = np.linspace(0, 1, len(request_states))

        else:
            priorities = [] 
            for request_state in request_states:
                instance_difficulty = request_state.instance.extra_data["difficulty"]
                priorities.append(
                    self._compute_negative_fisher_information(
                        model_ability=model_ability,
                        instance_difficulty=instance_difficulty,
                    )
                )

        # Add all requests to the queue with priority 0
        # Lower priority means being processed first
        request_state_queue = PriorityQueue()
        for request_state, priority in zip(request_states, priorities):
            request_state_queue.put(
                PrioritizedItem(
                    priority=priority,
                    request_state=request_state,
                )
            )

        return request_state_queue

    def _estimate_model_ability(
        self,
        old_ability: float,
        response_correctness: List[bool],
        instance_difficulties: List[float],
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ability = torch.tensor([old_ability], requires_grad=True, device=device)
        difficulties = torch.tensor(instance_difficulties, device=device)
        label = torch.tensor(response_correctness, device=device)

        optimizer = torch.optim.Adam([ability], lr=0.01)
        for _ in range(1000):
            optimizer.zero_grad()

            nll = torch.distributions.Bernoulli(
                logits=ability+difficulties
            ).log_prob(label).sum()

            loss = -nll
            loss.backward()
            optimizer.step()
        return ability.item()

    def run_execution_adaptive(
        self, 
        run_spec: RunSpec,
        request_states: List[RequestState],
    ):
        # Execute the requests in an adaptive manner
        model_ability = run_spec.adapter_spec.model_ability
        request_state_queue = self._construct_request_state_queue(
            request_states=request_states,
            model_ability=model_ability,
            priority="adaptive",
        )
        adaptive_request_states: List[RequestState] = []
        stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []
        adaptive_trajectory = {
            "model_ability": [],
            "response_correctness": [],
            "instance_difficulties": [],
        }

        for _ in tqdm(range(run_spec.adaptive_max_samples), desc="Adaptive execution"):
            pitem = request_state_queue.get()

            # Execute the request
            single_scenario_state, stat, per_instance_stat = self.run_execution(run_spec, [pitem.request_state])

            # Update the adaptive request states
            adaptive_request_states.extend(single_scenario_state.request_states)

            # Update the aggregated stats
            if len(stats) > 0:
                for idx, s in enumerate(stat):
                    stats[idx].merge(s)
            else:
                stats = stat

            # Update the per instance stats
            per_instance_stats.extend(per_instance_stat)

            # Update the adaptive trajectory
            adaptive_trajectory["model_ability"].append(model_ability)
            adaptive_trajectory["response_correctness"].append(
                float(per_instance_stat[0].stats[0].mean > 0.5)
            )
            adaptive_trajectory["instance_difficulties"].append(
                pitem.request_state.instance.extra_data["difficulty"]
            )

            # Estimate the model ability
            model_ability = self._estimate_model_ability(
                old_ability=model_ability,
                response_correctness=adaptive_trajectory["response_correctness"],
                instance_difficulties=adaptive_trajectory["instance_difficulties"],
            )

            # Update the priority
            request_state_queue = self._construct_request_state_queue(
                request_states=request_states,
                model_ability=model_ability,
                priority="adaptive",
            )

        # Create the scenario state
        scenario_state: ScenarioState = ScenarioState(
            adapter_spec=run_spec.adapter_spec,
            request_states=adaptive_request_states,
            annotator_specs=run_spec.annotators,
        )

        return scenario_state, stats, per_instance_stats, adaptive_trajectory