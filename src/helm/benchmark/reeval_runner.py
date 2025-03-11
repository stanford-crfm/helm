import dacite
import json
import os
import typing
from collections import Counter
from typing import Any, Dict, List
import torch

from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from helm.benchmark.scenarios.scenario import (
    EVAL_SPLITS,
    Scenario,
    create_scenario,
    Instance,
    get_scenario_cache_path,
    with_instance_ids,
    with_instance_difficulties,
)
from helm.benchmark.adaptation.adapters.adapter import Adapter
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.dry_run_metrics import DryRunMetric
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, create_metric, Stat
from helm.benchmark.runner import (
    Runner,
    RunSpec,
    remove_stats_nans,
    downsample_eval_instances,
    remove_per_instance_stats_nans,
)


class RelEffEvalRunner(Runner):
    """
    This runner implements the basic (non-amortized) method described in the paper
    `Reliable and Efficient Amortized Model-Based Evaluation`. This approach, which is
    also known as Computerized Adaptive Testing (CAT) within the framework of Item Response
    Theory (IRT), leverages adaptive testing to evaluate model performance.

    The difficulties of the questions are provided in a HuggingFace repository. In addition,
    the authors of the paper will supply a Python package for calculating these difficulties.
    At each iteration, the runner estimates the model's ability based on all previously
    administered questions and their corresponding responses. It then selects the next question
    whose difficulty is closest to the estimated ability, thereby optimally eliciting the
    model's ability.
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
        super().__init__(
            execution_spec=execution_spec,
            output_path=output_path,
            suite=suite,
            skip_instances=skip_instances,
            cache_instances=cache_instances,
            cache_instances_only=cache_instances_only,
            skip_completed_runs=skip_completed_runs,
            exit_on_error=exit_on_error,
        )

    def _estimate_model_ability(
        self,
        old_ability: float,
        response_correctness: List[bool],
        instance_difficulties: List[float],
    ) -> float:
        def closure():
            optim.zero_grad()
            probs = torch.sigmoid(ability - difficulties)
            loss = -torch.distributions.Bernoulli(probs=probs).log_prob(responses).mean()
            loss.backward()
            return loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        responses = torch.tensor(response_correctness, device=device)
        difficulties = torch.tensor(instance_difficulties, device=device)
        ability = torch.tensor([old_ability], requires_grad=True, device=device)
        optim = torch.optim.LBFGS([ability], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        for iteration in range(100):
            if iteration > 0:
                prev_ability = ability.clone()
                prev_loss = loss.clone()

            loss = optim.step(closure)

            if iteration > 0:
                d_loss = prev_loss - loss
                d_theta = torch.norm(prev_ability - ability, p=2)
                grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
                if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                    break

        return ability.item()

    def run_one(self, run_spec: RunSpec):
        run_path: str = self._get_run_path(run_spec)
        if self.skip_completed_runs and self._is_run_completed(run_path):
            hlog(f"Skipping run {run_spec.name} because run is completed and all output files exist.")
            return
        ensure_directory_exists(run_path)

        # TODO: is this necessary? Check reeval mode
        if self.executor.execution_spec.parallelism > 1:
            hlog("Reeval mode is not supported with parallelism > 1. Running with parallelism = 1.")
            raise RuntimeError("Reeval mode does not support parallelism > 1")

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

        # get difficulty information
        import numpy as np

        np.random.seed(42)
        difficulties = np.random.randn(len(instances)).tolist()
        if any([instance.extra_data is None for instance in instances]):
            instances = with_instance_difficulties(instances, difficulties)

        # Data preprocessing
        instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
            instances, self.executor.execution_spec.parallelism
        )

        # Adapt (convert to requests)
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)
        unasked_request_states: List[RequestState] = adapter.adapt(instances, self.executor.execution_spec.parallelism)

        # Execute the requests in an reeval manner
        model_ability = run_spec.adapter_spec.reeval_parameters.model_ability
        scenario_metric_name = run_spec.adapter_spec.reeval_parameters.metric_name

        asked_request_states: List[RequestState] = []
        stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []
        reeval_trajectory = {
            "model_ability": [],
            "response_correctness": [],
            "instance_difficulties": [],
        }

        for _ in tqdm(range(run_spec.adapter_spec.reeval_parameters.max_samples), desc="Reeval execution"):
            if not unasked_request_states:
                break

            selected_item = min(
                unasked_request_states, key=lambda item: abs(item.instance.extra_data["difficulty"] - model_ability)
            )
            unasked_request_states.remove(selected_item)

            # Execute the request
            single_scenario_state: ScenarioState = ScenarioState(
                adapter_spec=run_spec.adapter_spec,
                request_states=[selected_item],
                annotator_specs=run_spec.annotators,
            )

            # Execute (fill up results)
            single_scenario_state = self.executor.execute(single_scenario_state)

            # Annotate (post-process the results)
            single_scenario_state = self.annotator_executor.execute(single_scenario_state)

            # Apply the metrics
            # When performing a dry run, only estimate the number of tokens instead
            # of calculating the metrics.
            metrics: List[MetricInterface] = (
                [DryRunMetric()]
                if self.dry_run
                else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
            )
            stat: List[Stat] = []
            per_instance_stat: List[PerInstanceStats] = []
            with htrack_block(f"{len(metrics)} metrics"):
                for metric in metrics:
                    with htrack_block(metric):
                        metric_result: MetricResult = metric.evaluate(
                            single_scenario_state,
                            self.metric_service,
                            self.eval_cache_path,
                            self.executor.execution_spec.parallelism,
                        )
                        stat.extend(metric_result.aggregated_stats)
                        per_instance_stat.extend(metric_result.per_instance_stats)

            # Update the reeval request states
            asked_request_states.extend(single_scenario_state.request_states)

            # Update the aggregated stats
            if len(stats) > 0:
                for idx, s in enumerate(stat):
                    stats[idx].merge(s)
            else:
                stats = stat

            # Update the per instance stats
            per_instance_stats.extend(per_instance_stat)

            # Update the reeval trajectory
            reeval_trajectory["model_ability"].append(model_ability)
            scenario_metric_value = [s for s in per_instance_stat[0].stats if s.name.name == scenario_metric_name][
                0
            ].mean
            reeval_trajectory["response_correctness"].append(scenario_metric_value)
            reeval_trajectory["instance_difficulties"].append(selected_item.instance.extra_data["difficulty"])

            # Estimate the model ability
            model_ability = self._estimate_model_ability(
                old_ability=model_ability,
                response_correctness=reeval_trajectory["response_correctness"],
                instance_difficulties=reeval_trajectory["instance_difficulties"],
            )

        # Create the scenario state
        scenario_state: ScenarioState = ScenarioState(
            adapter_spec=run_spec.adapter_spec,
            request_states=asked_request_states,
            annotator_specs=run_spec.annotators,
        )

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

        write(
            os.path.join(run_path, "reeval_trajectory.json"),
            json.dumps(reeval_trajectory, indent=2),
        )

        cache_stats.print_status()
