from dataclasses import dataclass
from typing import List

from scenario import ScenarioSpec, create_scenario
from adapter import AdapterSpec, Adapter
from executor import ExecutionSpec, Executor
from metric import MetricSpec, create_metric


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to run on one scenario (apply the different adaptation procedures)
    """

    scenario: ScenarioSpec  # Which scenario
    adapter_spec: AdapterSpec  # Specifies how to adapt an instance into a set of requests
    metrics: List[MetricSpec]  # What to evaluate on


class Runner:
    """The main class for running everything."""

    def __init__(self, execution_spec: ExecutionSpec, run_specs: List[RunSpec]):
        self.executor = Executor(execution_spec)
        self.run_specs = run_specs

    def run_all(self):
        for run_spec in self.run_specs:
            self.run_one(run_spec)

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario = create_scenario(run_spec.scenario)

        # Build pieces (contextualized requests)
        adapter = Adapter(run_spec.adapter_spec)
        scenario_state = adapter.adapt(scenario)

        # Execute the requests
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        metrics = [create_metric(metric) for metric in run_spec.metrics]
        print(f"{len(metrics)} metrics")
        stats = []
        for metric in metrics:
            stats.extend(metric.evaluate(scenario_state))

        # Print out stats
        for stat in stats:
            print(stat)
