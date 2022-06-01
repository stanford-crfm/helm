from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, List, Tuple, Optional, cast

import pytrec_eval

from common.statistic import Stat, merge_stat
from .adapter import ScenarioState, RequestState
from .metric_service import MetricService
from .metric import Metric, MetricResult, PerInstanceStatsKey
from .metric_name import MetricName
from .scenario import VALID_SPLIT, MultipleRequestInstance


################################################################################
# Dataclass definitions                                                        #
################################################################################


@dataclass(frozen=True, eq=False)
class RequestStateGroup:
    """ Class representing a group of request states. """

    """ Unique ID for the request state group.  """
    group_id: str

    """ Instances of the group with ID group_id. """
    instances: List[MultipleRequestInstance]

    """ Request states corresponding to the instances. """
    request_states: List[RequestState]


################################################################################
# MultipleRequestMetric Functions                                              #
#                                                                              #
# Instructions for adding new functions operating on multiple request          #
# instances:                                                                   #
# > Ensure that the scenario you are writing your metric for uses the          #
#   MultipleRequestInstance instanve class instead of the vanilla instance     #
#   class.                                                                     #
# > Follow the signature for aggregated_runtime if adding new metric functions.#
# > Define MultipleRequestMetrics.{NEW_METRIC_NAME}_METRIC_NAME to be the name #
#   your function's name.                                                      #
# > Add "{NEW_METRIC_NAME}_METRIC_NAME": new_metric_function to                #
#   MultipleRequestMetrics.METRIC_NAME_TO_FUNCTION dictionary.                 #
################################################################################


def aggregated_runtime(
    request_state_groups: List[RequestStateGroup], train_trial_index
) -> Dict[PerInstanceStatsKey, List[Stat]]:
    """ Compute and return the aggregate runtime of running all the requests per group. """
    per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}
    for group in request_state_groups:
        if group.instances:
            run_time_stat = Stat(MetricName("runtime"))
            run_time_stat.add(sum([rs.result.request_time for rs in group.request_states if rs.result]))
            per_instance_stats[PerInstanceStatsKey(group.instances[0], train_trial_index)] = [run_time_stat]
    return per_instance_stats


################################################################################
# Metric Classes                                                               #
################################################################################


class MultipleRequestMetrics(Metric):
    """ Metrics for scenarios using the MultipleRequestInstance class.

    MultipleRequestInstance should be used when a scenario requires creating
    multiple instances to answer a single query.
    """

    """ Metrics supported. """
    AGGREGATED_RUNTIME_METRIC_NAME = "aggregated_runtime"

    """ Basic metrics for scenarios using the MultipleRequestInstance class. """
    BASIC_METRICS = [AGGREGATED_RUNTIME_METRIC_NAME]

    """ Dictionary mapping metric names to their respective functions. """
    METRIC_NAME_TO_FUNCTION: Dict[str, Callable] = {
        AGGREGATED_RUNTIME_METRIC_NAME: aggregated_runtime,
    }

    def __init__(self, metric_names: Optional[List[str]] = None, use_basic_metrics: bool = True):
        """ The constructor for the MultipleRequestMetrics class.

        Args:
            metric_names: The names of the metrics that will be run. The metric
                names must exist as keys in self.METRIC_NAME_TO_FUNCTION,
                case-sensitive.
            use_basic_metrics: If set, metrics in the self.BASIC_METRICS list
                will be added to the user provided set of metric_names.
        """
        # Dictionary mapping metric names to metrics
        self.metric_name_to_function = self.METRIC_NAME_TO_FUNCTION

        # Assign metric_names
        metric_names = metric_names if metric_names else []
        metric_names = metric_names + self.BASIC_METRICS if use_basic_metrics else metric_names
        for metric_name in metric_names:
            assert (
                metric_name in self.metric_name_to_function
            ), f"Metric name must be in {self.metric_name_to_function}, instead found {metric_name}."
        self.metric_names = list(set(metric_names))  # Remove duplicates

    @staticmethod
    def group_request_states(request_states: List[RequestState]) -> List[RequestStateGroup]:
        """ Group request_states using the group_id and return a list of groups. """
        group_dict: Dict[str, Dict[str, List[Any[MultipleRequestInstance, RequestState]]]] = defaultdict(
            lambda: {"instances": [], "request_states": []}
        )
        for rs in request_states:
            instance = cast(MultipleRequestInstance, rs.instance)
            if instance.group_id:
                group_dict[instance.group_id]["instances"].append(instance)
                group_dict[instance.group_id]["request_states"].append(rs)
        request_state_groups = [
            RequestStateGroup(group_id, d["instances"], d["request_states"]) for group_id, d in group_dict.items()
        ]
        return request_state_groups

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> MetricResult:
        """ Compute and return the metrics for self.metric_names. """
        aggregated_stats_dict: Dict[MetricName, Stat] = {}
        per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}
        # Loop through train trials
        for train_trial_index in range(scenario_state.adapter_spec.num_train_trials):
            validation_request_states = [rs for rs in scenario_state.request_states if rs.instance.split == VALID_SPLIT]
            request_state_groups = self.group_request_states(validation_request_states)
            # Run the request_state_groups through all metrics
            for metric_name in self.metric_names:
                # Get metric_per_instance_stats
                metric_function = self.metric_name_to_function[metric_name]
                metric_per_instance_stats = metric_function(request_state_groups, train_trial_index)
                per_instance_stats.update(metric_per_instance_stats)
                # Expand aggregate metrics
                for metric_stats in metric_per_instance_stats.values():
                    for metric_stat in metric_stats:
                        merge_stat(aggregated_stats_dict, metric_stat)
        return MetricResult(
            aggregated_stats=list(aggregated_stats_dict.values()), per_instance_stats=per_instance_stats
        )


class InformationRetrievalMetric(MultipleRequestMetrics):
    """ Information retrieval metrics. """

    """ Modes supported by this metric.

    There are multiple possible ways to formulate information retrieval
    problems as prompts. Following modes are supported by this metric:
        (1) binary_logprob: In binary_logprob mode, the information retrieval
            task is formulated as a binary classification problem. The model's
            task is to predict whether a given context contains an answer
            to a given query with appropriate amount of confidence. Different
            contexts for a query are then sorted using the model's output and
            its confidence in the output.
    """
    BINARY_LOGPROB_MODE = "binary_logprob"
    MODE_LIST = [BINARY_LOGPROB_MODE]

    """ Metric name for this class. """
    IR_METRIC_NAME = "information_retrieval"

    """ Token used to represent missing results.

    Results may be missing if the RequestState object didn't reach to a
    completed stage.
    """
    MISSING_RESULT_TOKEN = "missing"

    def __init__(
        self,
        measure_names: List[str],
        qrels_path: str,
        mode: str,
        correct_output: str,
        wrong_output: str,
        rank_limit: Optional[int] = None,
    ):
        """Constructor for the InformationRetrievalMetric.

        Args:
            measure_names: The trec_eval measure names that will be computed.
                Measure names must be in pytrec_eval.supported_measures.
            qrels_path: Path to the qrels file for the validation examples.
                Each row must follow the format below, where qid is the ID for
                the question, oid is the ID for the object and rel is the
                relevance of the said object for the said query:

                                <qid>   0   <oid>   <rel>
            mode: The information retrieval problem formulation used in the
                scenario implementing the information retrieval task. The mode
                must exists in self.MODES_LIST.
            correct_output: If the binary_logprob mode is selected, the string
                that should be outputted if the model predicts that the object
                given in the instance can answer the question.
            wrong_output: If the binary_logprob mode is selected, the string
                that should be outputted if the model predicts that the object
                given in the instance can not answer the question.
            rank_limit: The number of max top object rankings to keep for
                evaluation. This flag can be used, for example, if k
                validation examples were expanded by additional set examples,
                yet we still want to compute the scores for the top k objects
                ranked, irrespective of whether the objects came from the
                original k or the expanded set.
        """
        # Input validation
        assert mode in self.MODE_LIST, f"Mode must be one of {self.MODE_LIST}, instead found {mode}."
        self.mode = mode

        for measure_name in measure_names:
            assert (
                measure_name in pytrec_eval.supported_measures
            ), f"Measure name must be one of {pytrec_eval.supported_measures}, instead found {measure_name}."
        self.measure_names = measure_names

        self.correct_output = self.standardize(correct_output)
        self.wrong_output = self.standardize(wrong_output)

        if rank_limit:
            assert rank_limit >= 0
        self.rank_limit = rank_limit

        assert os.path.exists(qrels_path)
        self.qrels_path = qrels_path

        # Instance level variables we use throughout
        with open(qrels_path, "r") as f:
            self.qrels = pytrec_eval.parse_qrel(f)

        # Used to map get_run_ranking_dict to a function based on the selected mode
        self.mode_to_run_dict_function = {self.BINARY_LOGPROB_MODE: self.get_run_ranking_dict_binary_logprob_mode}
        self.get_run_ranking_dict_function = self.mode_to_run_dict_function[self.mode]

        # Initialize self.metric_name_to_function to include self.information_retrieval
        # Used in the evaluate method of the super class
        self.metric_name_to_function = {self.IR_METRIC_NAME: self.information_retrieval}
        self.metric_names = [self.IR_METRIC_NAME]

    @staticmethod
    def standardize(text: str) -> str:
        """ Strip and lower the given text. """
        return text.strip().lower()

    def triple_scorer(self, triple) -> Tuple[int, float]:
        """ Return a tuple score for the (oid, token, logprob) triple.

        Meant to be used as a key to the sorted function, which first compares
        triples using the first index of the returned tuple, and in case of a
        tie-break, uses the second index.
        """
        _, token, logprob = triple
        token_to_score = defaultdict(int, {self.wrong_output: 1, self.correct_output: 2})
        logprob_score = logprob if token == self.correct_output else -1 * logprob
        return (token_to_score[token], logprob_score)

    def get_sorted_oids(self, group: RequestStateGroup) -> List[str]:
        """ Return a list of sorted oids for the given group.  """
        triples = []
        for instance, request_state in zip(group.instances, group.request_states):
            oid = instance.request_id
            if oid:
                token, logprob = self.MISSING_RESULT_TOKEN, 0.0
                if request_state.result and request_state.result.completions:
                    token, logprob = (
                        request_state.result.completions[0].text,
                        request_state.result.completions[0].logprob,
                    )
                triples.append((oid, self.standardize(token), logprob))
        sorted_triples = sorted(triples, key=self.triple_scorer, reverse=True)
        return [oid for (oid, _, _) in sorted_triples]

    def get_run_ranking_dict_binary_logprob_mode(
        self, request_state_groups: List[RequestStateGroup]
    ) -> Dict[str, Dict[str, int]]:
        """ Compute a run dictionary from the request_state_groups.

        The returned run dictionary can be used as an input to evaluators of the
        pytrec_eval module.
        """
        run: Dict[str, Dict[str, int]] = {}
        for group in request_state_groups:
            sorted_pids = self.get_sorted_oids(group)
            sorted_pids = sorted_pids[: min(len(sorted_pids), self.rank_limit)] if self.rank_limit else sorted_pids
            run[group.group_id] = {pid: rank + 1 for rank, pid in enumerate(sorted_pids)}
        return run

    def information_retrieval(
        self, request_state_groups: List[RequestStateGroup], train_trial_index: int
    ) -> Dict[PerInstanceStatsKey, List[Stat]]:
        """ Compute the self.measures on the request_state_groups and return statistics. """
        run = self.get_run_ranking_dict_function(request_state_groups)
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, set(self.measure_names))
        evaluations: Dict[str, Dict[str, float]] = evaluator.evaluate(
            run
        )  # Dictionary mapping qid (group_id) to a dictionary of metric scores
        per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}
        for group in request_state_groups:
            if group.instances:
                group_stats = [
                    Stat(MetricName(metric_name)).add(score)
                    for metric_name, score in evaluations[group.group_id].items()
                ]
                per_instance_stats[PerInstanceStatsKey(group.instances[0], train_trial_index)] = group_stats
        return per_instance_stats
