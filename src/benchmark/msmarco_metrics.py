from typing import List, Dict, Tuple, Optional


from common.statistic import Stat
from .adapter import ScenarioState, RequestState
from .metric_service import MetricService
from .metric import Metric
from .scenario import VALID_SPLIT


class MSMARCOMetric(Metric):
    """Computes the metric for the MSMARCO datasets.

    Currently, the only supported metric is the `mean_reciprocal_rank` metric.
    Read the documentation in the `reciprocal_rank` method for details on
    how this is computed.
    """

    # Names of the metrics of this class
    METRIC_NAMES = ["mean_reciprocal_rank"]

    # Constants for the answer texts
    YES_ANSWER = "yes"
    NO_ANSWER = "no"

    def __init__(self, name: str, topk_list: Optional[List[int]] = None):
        """Constructor for the MSMARCOMetric.

        Args:
            name: The metric to be used. Should be one of self.METRIC_NAMES.
            topk_list: The number of top rankings we should consider when computing
                the reciprocal ranks. For example, if topk_list=[5, 10], we will
                compute two stats, one considering the top 5 ranking, while the
                other ones considers the top 10 rankings.
        """

        # Set the name of the metric
        if name not in self.METRIC_NAMES:
            raise ValueError(f"Expected name to be one of: [{', '.join(self.METRIC_NAMES)}]")
        self.name = name

        # Helpers for calling the correct metric
        self.METRIC_FNS = {"mean_reciprocal_rank": self.reciprocal_rank}
        self.metric_fn = self.METRIC_FNS[name]

        # Topk list for the MRR metric (MRR@10, MRR@20 etc.)
        if not topk_list:
            topk_list = [10]
        self.topk_list = topk_list

    @staticmethod
    def flatten_id(instance_id: str) -> tuple[int, int, bool]:
        """Gets fields from an instance_id.

        Instance id is of the following form:

            "qid_{qid}-pid_{pid}-gold_{gold}"

        Returns:
            qid: query id.
            pid: passage id.
            gold: Whether the instance corresponds to a gold example.
        """
        qid_str, pid_str, gold_str = instance_id.split("-")
        qid = int(qid_str.split("_")[1])
        pid = int(pid_str.split("_")[1])
        gold = bool(gold_str.split("_")[1])
        return qid, pid, gold

    def get_answer(self, is_yes: bool) -> str:
        """Returns the answer text given whether it is the yes answer.

        Args:
            is_yes: Whether the answer should be a yes answer.

        Returns:
            answer: "yes" or "no" depending on is_yes.
        """
        if is_yes:
            return self.YES_ANSWER
        return self.NO_ANSWER

    def get_qid_dictionaries(
        self, request_states: List[RequestState]
    ) -> Tuple[Dict[int, int], Dict[str, List[Tuple[int, int, float]]]]:
        """Extracts qid_to_gold_pid and qid_pid_logprob_dict dictionaries from the request_states.

        Args:
            request_states: List of request states.

        Returns:
            qid_to_gold_pid: Dictionary mapping a qid to the gold pid match.
            qid_pid_logprob_dict: Dictionary containing the following:

                {
                    self.YES_ANSWER: qid_pid_logprob_list,
                    self.NO_ANSWER: qid_pid_logprob_list
                }

                Where qid_pid_logprob_list is a list containing tuples of the form:

                    (qid, pid, logprob)
        """
        # Initialize helper dictionaries
        qid_to_gold_pid = {}
        qid_pid_logprob_dict = {self.YES_ANSWER: list(), self.NO_ANSWER: list()}
        # Iterate through the request states
        for rs in request_states:
            # Extract important information from the ID
            instance_id = rs.instance.id
            qid, pid, is_gold = self.flatten_id(instance_id)
            # If this is the gold example, add it to our gold list
            if rs.result.completions:
                # Populate the gold mapping dictionary
                if is_gold:
                    qid_to_gold_pid[qid] = pid
                # Get the answer from the model
                is_yes = (is_gold and rs.result.success) or (not is_gold and not rs.result.success)
                answer = self.get_answer(is_yes)
                qid_pid_logprob_dict[answer].append((qid, pid, rs.result.completions[0].logprob))

        return qid_to_gold_pid, qid_pid_logprob_dict

    def get_ranked_pid_list(self, qid_pid_logprob_dict: Dict[str, List[Tuple[int, int, float]]], qid: int) -> List[int]:
        """Returns the ranked pid list for a given qid.

        Args:
            qid_pid_logprob_dict: Dictionary containing the following:

                {
                    self.YES_ANSWER: qid_pid_logprob_list,
                    self.NO_ANSWER: qid_pid_logprob_list
                }

                Where qid_pid_logprob_list is a list containing tuples of the form:

                    (qid, pid, logprob)
            qid: ID for the query.

        Returns:
            ranked_pids: List of pids, ranked based on their match to the query with the qid as judged
                by the language model.
        """
        # Take the examples where the model answered yes, and sort them from the biggest to the lowest
        yes_tuples = [t for t in qid_pid_logprob_dict[self.YES_ANSWER] if t[0] == qid]
        yes_tuples = sorted(yes_tuples, key=lambda t: t[2], reverse=True)
        ranked_pids_yes = [t[1] for t in yes_tuples]

        # Take the examples where the model answered yes, and sort them from the lowest to the biggest
        no_tuples = [t for t in qid_pid_logprob_dict[self.NO_ANSWER] if t[0] == qid]
        no_tuples = sorted(no_tuples, key=lambda t: t[2], reverse=True)
        ranked_pids_no = [t[1] for t in no_tuples]

        # We combine the two lists, putting the yes answers to the front
        ranked_pids = ranked_pids_yes + ranked_pids_no

        return ranked_pids

    def reciprocal_rank(self, scenario_state: ScenarioState, metric_service: MetricService) -> List[Stat]:
        """Computes the mean reciprocal ranks of the model's ranking of the golden passages.

        Note that there is only one instance where we ask the model whether a passage
        with the ID pid contains the query with the ID qid. For each qid, we
        assign a ranking to each pid that we queried the model with as follows:
        - We get the model's answer, "yes" or "no", and the logprob of the answer
            for each pid.
        - We rank the answers we got using the following scheme:
            High => "yes", high logprob
                 => "yes", low  logprob
                 => "no",  low  logprob
            Low  => "no",  high logprob
        - For each k in self.topk_list:
            * We check whether the pid that corresponds to the golden passage for
                a given query with qid is ranked at most k.
            * If so, we set the reciprocal rank to be 1/ranking.
            * Else, we set the reciprocal rank to be 0.
            * We add the value we found to a Stat with name "RR@k", depending on
                the value of k.

        Returns:
            topk_stats: List of "RR@k" stats. One Stat will be computed for every
                k value in the self.topk_list. For example, if self.topk_list is
                [10, 20] the returned list will be [Stat(name="RR@10), Stat(name="RR@20)].
        """
        adapter_spec = scenario_state.adapter_spec
        topk_to_stat = {k: Stat(name=f"RR@{k}") for k in self.topk_list}

        # Iterate through trials
        for train_trial_index in range(adapter_spec.num_train_trials):

            validation_request_states = [rs for rs in scenario_state.request_states if rs.instance.split == VALID_SPLIT]
            qid_to_gold_pid, qid_pid_logprob_dict = self.get_qid_dictionaries(validation_request_states)

            for qid, gold_pid in qid_to_gold_pid.items():
                # Rank the pids for the qid
                ranked_pids = self.get_ranked_pid_list(qid_pid_logprob_dict, qid)

                # Get the rank of the gold
                try:
                    rank_gold = ranked_pids.index(gold_pid) + 1
                except ValueError:
                    rank_gold = None

                # Compute MRR
                for k, stat in topk_to_stat.items():
                    rr = 0
                    if rank_gold and rank_gold <= k:
                        rr = 1 / float(rank_gold)
                    stat.add(rr)

        return list(topk_to_stat.values())

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> List[Stat]:
        """Returns stats for the MSMARCO metric.
        """
        rr_stats = self.metric_fn(scenario_state, metric_service)
        return rr_stats
