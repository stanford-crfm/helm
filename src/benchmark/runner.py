import json
import os
from dataclasses import dataclass, asdict, replace
from typing import List, Optional
import uuid
import datetime

from tqdm import tqdm
from sqlitedict import SqliteDict

from common.general import ensure_directory_exists, write, write_lines, pickle, unpickle, UUIDEncoder
from common.hierarchical_logger import hlog, htrack_block
from common.request import RequestResult
from proxy.remote_service import RemoteService
from .adapter_service import AdapterService
from .augmentations.data_augmenter import DataAugmenterSpec
from .metric_service import MetricService
from .scenario import Scenario, ScenarioSpec, create_scenario, Instance
from .adapter import (
    AdapterSpec,
    Adapter,
    InteractionRound,
    InteractionTrace,
    InteractiveAdapter,
    InteractiveAdapterSpec,
    RequestState,
    ScenarioState,
    UserInput,
    create_interactive_adapter,
    Survey,
)
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor, render_request_state
from .metric import Metric, MetricSpec, create_metric, Stat
from .tokens_metric import TokensMetric


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of metrics.
    """

    # Unique identifier of the RunSpec
    name: str

    # Which scenario
    scenario: ScenarioSpec

    # Specifies how to adapt an instance into a set of requests
    adapter_spec: AdapterSpec

    # What to evaluate on
    metrics: List[MetricSpec]

    # Data augmenter. The default `DataAugmenterSpec` does nothing.
    data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec()

    # Adapter for interactive scenarios
    interactive_adapter: Optional[InteractiveAdapterSpec] = None


class Runner:
    """
    The main entry point for running the entire benchmark.  Mostly just
    dispatches to other classes.
    """

    def __init__(
        self,
        execution_spec: ExecutionSpec,
        output_path: str,
        run_specs: List[RunSpec],
        skip_instances: bool,
        pre_interaction: bool,
        post_interaction: bool,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run = execution_spec.dry_run
        self.adapter_service = AdapterService(self.executor.remote_service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.remote_service, execution_spec.auth)
        self.output_path = output_path
        self.run_specs = run_specs
        self.skip_instances = skip_instances
        self.pre_interaction = pre_interaction
        self.post_interaction = post_interaction
        assert not (
            self.pre_interaction and self.post_interaction
        ), "Both pre- and- post interaction cannot be simultaneously set"
        ensure_directory_exists(self.output_path)

    def run_all(self):
        for run_spec in self.run_specs:
            with htrack_block(f"Running {run_spec.name}"):
                if run_spec.adapter_spec.interactive:
                    assert (
                        self.pre_interaction or self.post_interaction
                    ), "For interactive scenarios, either pre- or post- interaction needs to be set"
                    if self.pre_interaction:
                        self.pre_execute(run_spec, write_state=True)
                    else:
                        runs_path = os.path.join(self.output_path, "runs", run_spec.name)
                        ensure_directory_exists(runs_path)
                        self.post_execute(run_spec, runs_path)  # Scenario state is read from the path
                else:
                    self.run_one(run_spec)

    def pre_execute(self, run_spec: RunSpec, write_state=False):
        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario)

        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenarios_path = os.path.join(self.output_path, "scenarios")
        ensure_directory_exists(scenarios_path)
        scenario.output_path = os.path.join(scenarios_path, scenario.name)
        scenario.definition_path = scenario.get_definition_path()
        ensure_directory_exists(scenario.output_path)
        runs_path = os.path.join(self.output_path, "runs", run_spec.name)
        ensure_directory_exists(runs_path)

        # Data preprocessing
        if not self.skip_instances:
            instances: List[Instance] = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(scenario)
        else:
            instances = []

        # Adaptation
        adapter = Adapter(run_spec.adapter_spec, self.adapter_service)
        scenario_state: ScenarioState = adapter.adapt(instances)

        # Output benchmarking information and results to files
        write(os.path.join(runs_path, "run_spec.json"), json.dumps(asdict(run_spec), indent=2))
        pickle(os.path.join(runs_path, "run_spec.pkl"), run_spec)

        scenario_dict = asdict(scenario)
        scenario_dict["instances"] = [asdict(instance) for instance in scenario_state.instances]
        write_lines(os.path.join(runs_path, "scenario.txt"), scenario.render_lines(scenario_state.instances))
        write(os.path.join(runs_path, "scenario.json"), json.dumps(scenario_dict, indent=2))

        if write_state:
            write_lines(os.path.join(runs_path, "scenario_state.txt"), scenario_state.render_lines())
            write(
                os.path.join(runs_path, "scenario_state.json"),
                json.dumps(asdict(scenario_state), indent=2, cls=UUIDEncoder),
            )
            pickle(os.path.join(runs_path, "scenario_state.pkl"), scenario_state)
            print(scenario_state)
            if scenario_state.interaction_traces:
                with SqliteDict(
                    os.path.join(runs_path, "interaction_traces.sqlite"), tablename="interaction_traces", flag="n"
                ) as trace_db:
                    for interaction_trace in scenario_state.interaction_traces:
                        trace_db[str(interaction_trace._id)] = interaction_trace
                    trace_db.commit()

        return scenario, runs_path, scenario_state

    def execute_one(self, runs_path: str, scenario_state: ScenarioState):
        # Execution
        scenario_state = self.executor.execute(scenario_state)

        write_lines(os.path.join(runs_path, "scenario_state.txt"), scenario_state.render_lines())
        write(os.path.join(runs_path, "scenario_state.json"), json.dumps(asdict(scenario_state), indent=2))
        pickle(os.path.join(runs_path, "scenario_state.pkl"), scenario_state)

    def post_execute(self, run_spec: RunSpec, runs_path: str, scenario_state: Optional[ScenarioState] = None):
        if scenario_state is None:
            loaded_scenario_state: ScenarioState = unpickle(os.path.join(runs_path, "scenario_state.pkl"))

            # Load interaction traces from the sqlite database
            if loaded_scenario_state.interaction_traces is not None:
                with SqliteDict(
                    os.path.join(runs_path, "interaction_traces.sqlite"), tablename="interaction_traces"
                ) as trace_db:
                    loaded_scenario_state.interaction_traces = [
                        trace_db[str(interaction_trace._id)]
                        for interaction_trace in loaded_scenario_state.interaction_traces
                    ]
                loaded_scenario_state.__post_init__()

            scenario_state = loaded_scenario_state
        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = ([] if self.dry_run else [create_metric(metric) for metric in run_spec.metrics]) + [
            TokensMetric()
        ]
        stats: List[Stat] = []
        with htrack_block(f"{len(metrics)} metrics"):
            for metric in metrics:
                with htrack_block(metric):
                    stats.extend(metric.evaluate(scenario_state, self.metric_service))

        # Print out stats
        with htrack_block("Stats"):
            for stat in stats:
                hlog(stat)

        write_lines(os.path.join(runs_path, "metrics.txt"), [str(stat) for stat in stats])
        write(os.path.join(runs_path, "metrics.json"), json.dumps([asdict(stat) for stat in stats], indent=2))

    def run_one(self, run_spec: RunSpec):
        # Load scenario, create directories, adapt and write run_spec, scenario, scenario_state
        _, runs_path, scenario_state = self.pre_execute(run_spec)

        # Execution
        scenario_state = self.execute_one(runs_path, scenario_state)

        # Run metrics and write
        self.post_execute(run_spec, runs_path, scenario_state)


class InteractiveRunner:
    """
    The `InteractiveRunner` operates on a persisted `ScenarioState` which has a bunch of incomplete interaction traces.
    In the event of a user interaction, it reads the current state for that instance,
    creates and executes a request to the lm, updates the interaction state on the disk and responds to the user.

    It assumes that two threads/processes do not attempt to process the same instance concurrently,
    as it could lead to data loss.
    """

    def __init__(self, execution_spec: ExecutionSpec, output_path: str, run_spec: RunSpec):
        self.execution_spec = execution_spec
        self.remote_service = RemoteService(self.execution_spec.url)
        # self.executor: Executor = Executor(execution_spec)
        # self.adapter_service = AdapterService(self.executor.remote_service, execution_spec.auth)
        self.output_path = output_path
        ensure_directory_exists(self.output_path)
        self.run_spec = run_spec
        self.runs_path = os.path.join(self.output_path, "runs", run_spec.name)
        ensure_directory_exists(self.runs_path)
        assert self.run_spec.interactive_adapter is not None, "Need an InteractiveAdapterSpec for InteractiveRunner"
        self.interactive_adapter: InteractiveAdapter = create_interactive_adapter(
            interactive_adapter_spec=self.run_spec.interactive_adapter
        )

    def process(self, state: RequestState) -> RequestState:
        result: RequestResult = self.remote_service.make_request(self.execution_spec.auth, state.request)
        state = replace(state, result=result)
        tqdm.write(render_request_state(state))
        return state

    def load_interaction_trace(self, interaction_trace_id: uuid.UUID) -> InteractionTrace:
        with SqliteDict(
            os.path.join(self.runs_path, "interaction_traces.sqlite"), tablename="interaction_traces"
        ) as trace_db:
            interaction_trace = trace_db[str(interaction_trace_id)]
        return interaction_trace

    def save_interaction_trace(self, interaction_trace: InteractionTrace) -> None:
        with SqliteDict(
            os.path.join(self.runs_path, "interaction_traces.sqlite"), tablename="interaction_traces"
        ) as trace_db:
            interaction_trace_id = str(interaction_trace._id)
            trace_db[interaction_trace_id] = interaction_trace
            trace_db.commit()

    def initialize_interaction_trace(self, user_id: str, interaction_trace_id: uuid.UUID) -> InteractionTrace:
        interaction_trace = self.load_interaction_trace(interaction_trace_id=interaction_trace_id)
        interaction_trace.user_id = user_id
        assert (
            len(interaction_trace.trace) > 0
        ), "InteractionTrace.trace should have at least pre-filled InteractionRound"

        if (
            len(interaction_trace.trace) > 1
            or interaction_trace.trace[0].user_input is not None
            or interaction_trace.trace_completed
        ):
            # The interaction_trace already has user_input
            # Do not overwrite
            return interaction_trace

        # Postprocess the initial request
        first_interaction_round = interaction_trace.trace[0]
        assert first_interaction_round.user_input is None
        new_request_state = self.interactive_adapter.postprocess_initial_request(
            first_interaction_round.request_state, self.run_spec.adapter_spec
        )
        first_interaction_round = InteractionRound(
            user_input=None, request_state=new_request_state, time=datetime.datetime.now()
        )
        if self.interactive_adapter.user_initiated is False:
            new_request_state = self.interactive_adapter.initial_lm_request(first_interaction_round.request_state)
            new_request_state = self.process(new_request_state)
            hlog(new_request_state.render_lines())
            first_interaction_round = InteractionRound(user_input=None, request_state=new_request_state)

        # Handle toxicity, if any
        clean_output, generated_toxic = self.interactive_adapter.filter_toxic_generations(first_interaction_round)
        if generated_toxic and first_interaction_round.request_state.result is not None:
            toxic_generation = first_interaction_round.request_state.result.completions[
                0
            ].text  # Original generation was toxic
            first_interaction_round = self.replace_toxic_with_safe(first_interaction_round, clean_output)
            interaction_trace.toxic_generations.append(toxic_generation)
        interaction_trace.trace[0] = first_interaction_round

        self.save_interaction_trace(interaction_trace=interaction_trace)
        return interaction_trace

    def handle_user_input(self, interaction_trace_id: uuid.UUID, user_input: UserInput) -> RequestResult:
        """Handle user input, get LM response, save and return it"""
        interaction_trace = self.load_interaction_trace(interaction_trace_id=interaction_trace_id)
        assert not interaction_trace.trace_completed, "Cannot add new user inputs to a completed trace"

        new_request_state: RequestState = self.interactive_adapter.adapt_user_input(interaction_trace, user_input)
        new_request_state = self.process(new_request_state)
        print(new_request_state.render_lines())
        assert new_request_state.result

        # Handle toxicity
        interaction_round = InteractionRound(
            user_input=user_input, request_state=new_request_state, time=datetime.datetime.now()
        )
        clean_output, generated_toxic = self.interactive_adapter.filter_toxic_generations(interaction_round)
        if generated_toxic and interaction_round.request_state.result is not None:
            toxic_generation = interaction_round.request_state.result.completions[
                0
            ].text  # Original generation was toxic
            interaction_round = self.replace_toxic_with_safe(interaction_round, clean_output)
            interaction_trace.toxic_generations.append(toxic_generation)

        interaction_trace.trace.append(interaction_round)
        print(interaction_trace.render_lines())

        self.save_interaction_trace(interaction_trace=interaction_trace)

        return new_request_state.result

    def replace_toxic_with_safe(self, toxic_round: InteractionRound, clean_text: str) -> InteractionRound:
        """Replace toxic generation with clean generation in InteractionRound"""
        assert toxic_round.request_state.result is not None
        clean_completions = toxic_round.request_state.result.completions
        clean_completions[0] = replace(clean_completions[0], text=clean_text)
        clean_result = replace(toxic_round.request_state.result, completions=clean_completions)
        clean_request_state = replace(toxic_round.request_state, result=clean_result)
        clean_round = replace(toxic_round, request_state=clean_request_state)
        clean_round = replace(clean_round, generated_toxic=True)
        return clean_round

    def terminate_interaction_trace(self, interaction_trace_id: uuid.UUID):
        interaction_trace = self.load_interaction_trace(interaction_trace_id=interaction_trace_id)
        interaction_trace.trace_completed = True
        self.save_interaction_trace(interaction_trace=interaction_trace)

    def handle_survey(self, user_id: str, interaction_trace_id: uuid.UUID, survey):
        """Store the result of a survey after an interaction"""

        interaction_trace = self.load_interaction_trace(interaction_trace_id=interaction_trace_id)
        assert all(
            s.user_id != user_id for s in interaction_trace.surveys
        ), "Survey exists for the given used_id, cannot overwrite"
        survey = Survey(user_id=user_id, data=survey)
        interaction_trace.surveys.append(survey)
        self.save_interaction_trace(interaction_trace=interaction_trace)
