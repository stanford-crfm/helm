from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pytrec_eval

from common.request import RequestResult
from ..adapter import AdapterSpec, RequestState, CORRECT_TAG, ADAPT_RANKING_BINARY
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat


@dataclass
class RankingObject:
    """ Simple dataclass used to store features used for ranking objects. """

    """ Reference index corresponding to this object. """
    reference_index: int

    """ Whether the object corresponds to a gold object. """
    gold: bool = False

    """ Object text. """
    text: Optional[str] = None

    """ True relevance. """
    relevance: Optional[int] = None

    """ Top-k rank. """
    rank: Optional[int] = None

    """ Model output for the object. """
    model_output: Optional[str] = None

    """ Model logprob for the completion. """
    model_logprob: Optional[float] = None

    """ Model relevance.

    Relevance of this object for the query as determined by the model output
    and logprob.
    """
    model_relevance: Optional[int] = None


class RankingMetric(Metric):
    """ Ranking metric. """

    """ Modes supported by this metric.

    Following modes are supported by this metric:
        (1) ADAPT_RANKING_BINARY: In binary_ranking mode, the model's task is to
            predict whether a given context contains an answer to a given query.
            Different contexts for a query are then sorted using the label
            outputted by model as well as the model's confidence in the label.
    """
    MODE_LIST = [ADAPT_RANKING_BINARY]

    """ Measures. """
    RECIP_RANK_MEASURE = "recip_rank"
    SUCCESS_MEASURE = "success"
    RECALL_MEASURE = "recall"
    NDCG_CUT_MEASURE = "ndcg_cut"
    SUPPORTED_MEASURES = [RECIP_RANK_MEASURE, SUCCESS_MEASURE, RECALL_MEASURE, NDCG_CUT_MEASURE]
    BINARY_MEASURES = [RECIP_RANK_MEASURE, RECALL_MEASURE, SUCCESS_MEASURE]
    CUSTOM_PARAMETRIZED_MEASURES = [RECIP_RANK_MEASURE]

    """ Token used to represent missing results.

    Results may be missing if the RequestState object didn't reach to a
    completed stage.
    """
    MISSING_RESULT_TEXT = "missing"

    def __init__(
        self,
        mode: str,
        measure_names: List[str],
        correct_output: str,
        wrong_output: str,
        rank: Optional[int] = None,
        multiple_relevance_values: bool = False,
    ):
        """Constructor for the RankingMetric.

        Args:
            mode: The adaptation mode used. The mode must exists in
                self.MODES_LIST.
            measure_names: The trec_eval measure names that will be computed.
                Measure names must be measure names supported by the official
                trec_eval measure. List of supported measures can be found in
                self.SUPPORTED_MEASURES. Note that:
                    (1) We also accept the parametrized versions
                        (e.g. "measure_name.k") of self.SUPPORTED_MEASURES
                        measures.
                    (2) We accept any measure that's in either "measure_name" or
                        "measure_name.k" form, where measure_name is in
                        pytrec_eval.supported_measures, but note that
                        self.BINARY_MEASURES list must be modified to
                        include any new binary measures.
            correct_output: If the ADAPT_RANKING_BINARY mode is selected,
                the string that should be outputted if the model predicts that
                the object given in the instance can answer the question.
            wrong_output: If the ADAPT_RANKING_BINARY mode is selected, the
                string that should be outputted if the model predicts that the
                object given in the instance can not answer the question.
            rank: The optional number of max top document rankings to keep for
                evaluation. If None, all the rankings are evaluated. If topk is
                specified, only the top min(len(rankings), topk) document
                rankings are kept and used for evaluation for each query.
            multiple_relevance_values: Query relevance values can either be
                binary or take on multiple values, as explained below. This flag
                indicates whether the relevance values can take multiple values.
                    (1) Binary relevance values: If the relevance values are
                        binary, it means that all the matching relationships
                        would get assigned a relevance value of 1, while the
                        known non-matching relationships would get assigned a
                        relevance value of 0.
                    (2) Multiple relevance values: In the case of multiple
                        relevance values, the value of 0 will be interpreted as
                        non-matching relationship, but any other value would be
                        interpreted as a matching relationship differing
                        strengths.
        """
        # Input validation
        assert mode in self.MODE_LIST, f"Mode must be one of {self.MODE_LIST}, instead found {mode}."
        self.mode = mode

        for measure_name in measure_names:
            assert self.validate_measure_name(measure_name), f"""Measure name {measure_name} isn't supported."""
        self.measure_names = measure_names

        self.correct_output = correct_output
        self.wrong_output = wrong_output

        if rank:
            assert rank >= 0
        self.rank = rank

        self.multiple_relevance_values = multiple_relevance_values

        # Decide which model relevance function to use
        MODE_TO_RELEVANCE_FUNCTION = {ADAPT_RANKING_BINARY: self.compute_model_relevance_binary_ranking}
        self.model_relevance_fn = MODE_TO_RELEVANCE_FUNCTION[self.mode]

    @staticmethod
    def parse_measure_name(measure_name) -> Tuple[str, Optional[int]]:
        """ Parse measure name of the form 'measure_name' or 'measure_name.k' into its parts. """
        measure_delimiter = "."
        if measure_delimiter in measure_name:
            measure_name, k = measure_name.split(measure_delimiter)
            return measure_name, int(k)
        return measure_name, None

    def validate_measure_name(self, measure_name: str) -> bool:
        """ Check if the provided measure_name is a valid measure name. """
        measure_name, k = self.parse_measure_name(measure_name)
        if measure_name in self.SUPPORTED_MEASURES + list(pytrec_eval.supported_measures):
            return True
        return False

    def measure_to_metric_name(self, measure_name: str) -> str:
        """ Convert the measure name to an easier to read metric name. """
        MEASURE_TO_METRIC_NAME = {"ndcg_cut": "NDCG", "recip_rank": "RR"}
        parsed_measure_name, k = self.parse_measure_name(measure_name)
        metric_name_str = MEASURE_TO_METRIC_NAME.get(parsed_measure_name, parsed_measure_name.capitalize())
        k_str = f"@{k}" if k else ""
        return f"{metric_name_str}{k_str}"

    @staticmethod
    def standardize(text: str) -> str:
        """ Strip and lower the given text. """
        return text.strip().lower()

    @staticmethod
    def get_tag_value(tag) -> str:
        _, value = tag.split("=")
        return value

    def get_run_relevances(self, ranking_objs: List[RankingObject], rank_limit: Optional[int] = None) -> Dict[int, int]:
        """ Get the relevance dictionary for the run. """
        assert all([r.model_relevance is not None for r in ranking_objs])
        if rank_limit:
            assert all([r.rank is not None for r in ranking_objs])
            return {r.reference_index: r.model_relevance for r in ranking_objs if r.rank <= rank_limit}  # type: ignore
        return {r.reference_index: r.model_relevance for r in ranking_objs}  # type: ignore

    def get_true_relevances(self, ranking_objects: List[RankingObject]) -> Dict[int, int]:
        """ Get the true relevance dictionary. """
        assert all([obj.relevance is not None for obj in ranking_objects])
        return {obj.reference_index: obj.relevance for obj in ranking_objects if obj.relevance}

    def object_to_score_binary_ranking(self, ranking_object: RankingObject) -> Tuple[int, float]:
        """ Score the given ranking object. """
        # Input validation
        assert ranking_object.model_output is not None
        token = self.standardize(ranking_object.model_output)
        assert ranking_object.model_logprob is not None
        logprob = ranking_object.model_logprob

        # Score
        if token == self.standardize(self.correct_output):
            return 2, logprob
        elif token == self.standardize(self.wrong_output):
            return 1, -1 * logprob
        else:
            return 0, -1 * logprob

    def compute_model_relevance_binary_ranking(self, ranking_objects: List[RankingObject]) -> List[RankingObject]:
        """ Populate the model_relevance field for the ranking objects. """
        # Sort the ranking objects
        ranking_objects_sorted = sorted(ranking_objects, key=self.object_to_score_binary_ranking, reverse=True)

        # Add model relevance
        for ind, ranking_object in enumerate(ranking_objects_sorted):
            ranking_object.model_relevance = len(ranking_objects_sorted) - ind

        return ranking_objects_sorted

    def get_model_output_logprob(self, result: RequestResult) -> Tuple[str, float]:
        """ Get model's completion and the relevant logprob from a request state. """
        model_output, model_logprob = self.MISSING_RESULT_TEXT, 0.0
        if result and result.completions:
            # TODO: Decide how to pick between multiple completions.
            if result.completions[0].tokens:
                token = result.completions[0].tokens[0]
                model_output, model_logprob = token.text, token.logprob
        return model_output, model_logprob

    def create_ranking_object(self, request_state: RequestState) -> RankingObject:
        """ Create a RankingObject from a RequestState. """
        # Get reference index
        assert request_state.reference_index is not None
        reference_index = request_state.reference_index
        reference = request_state.instance.references[request_state.reference_index]

        # Get gold, relevance and rank fields
        gold = CORRECT_TAG in reference.tags
        relevance, rank = None, None
        for tag in reference.tags:
            if "relevance" in tag:
                relevance = int(self.get_tag_value(tag))
            elif "rank" in tag:
                rank = int(self.get_tag_value(tag))

        # Get model completion and the associated log probability
        model_output, model_logprob = self.get_model_output_logprob(request_state.result)

        # Create a RankingObject
        ranking_object = RankingObject(
            reference_index=reference_index,
            text=reference.output,
            gold=gold,
            relevance=relevance,
            rank=rank,
            model_output=model_output,
            model_logprob=model_logprob,
        )
        return ranking_object

    @staticmethod
    def limit_relevance_dict(relevance_dict: Dict[int, int], k: int) -> Dict[int, int]:
        """ Return a dictionary containing the k pairs with the highest relevance values. """
        # Sort the dictionary by the relevance values and limit the number of elements.
        relevance_dict = {k: v for k, v in sorted(relevance_dict.items(), key=lambda item: item[1], reverse=True)}
        limit = min(len(relevance_dict), k)  # Limit the dictionary
        relevance_dict = {key: relevance_dict[key] for key in list(relevance_dict.keys())[:limit]}
        return relevance_dict

    @staticmethod
    def binarize_relevance_dict(d: Dict[int, int]) -> Dict[int, int]:
        """ Binarize the dict by setting the values that are 1 to 0.

        Values greater than 1 stay untouched.
        """
        return {k: 0 if v == 1 else v for k, v in d.items()}

    def compute_measure(
        self, true_relevances: Dict[int, int], run_relevances: Dict[int, int], evaluator_measure_name: str
    ) -> float:
        """ Compute measure_name on the true and run relevances provided.

        Return a dictionary mapping metric names to the computed values.
        """
        # Parse the measure name
        parsed_measure_name, k = self.parse_measure_name(evaluator_measure_name)

        # If the measure is a binary measure, we binarize the true relevance values.
        if parsed_measure_name in self.BINARY_MEASURES and self.multiple_relevance_values:
            true_relevances = self.binarize_relevance_dict(true_relevances)

        # pytrec doesn't support parametrization for some measure names. For
        # such measures, we achieve parametrization by simply limiting
        # run_relevances dictionary to the top k relevances.
        if parsed_measure_name in self.CUSTOM_PARAMETRIZED_MEASURES and k:
            evaluator_measure_name = parsed_measure_name  # Pass the non-parametrized name to pytrec
            run_relevances = self.limit_relevance_dict(run_relevances, k)

        # RelevanceEvaluator expects a dictionary that maps Query IDs to their
        # relevance dictionaries. To pass our relevance dictionaries in the
        # right format, we wrap them in a dictionary.
        measure_names, true_qrels, run_qrels = {evaluator_measure_name}, {"_": true_relevances}, {"_": run_relevances}
        evaluator = pytrec_eval.RelevanceEvaluator(true_qrels, measure_names)
        evaluations = evaluator.evaluate(run_qrels)
        score = list(evaluations["_"].values())[0]  # TODO
        return score

    def compute_measures(
        self, true_relevances: Dict[int, int], run_relevances: Dict[int, int], metric_name_suffix=""
    ) -> List[Stat]:
        """ Compute self.measures on the true and run relevances provided. """
        # Loop through the measure names and populate the stats list.
        stats: List[Stat] = []
        for evaluator_measure_name in self.measure_names:
            score = self.compute_measure(true_relevances, run_relevances, evaluator_measure_name)
            metric_name = self.measure_to_metric_name(evaluator_measure_name)
            stat = Stat(MetricName(metric_name + metric_name_suffix)).add(score)
            stats.append(stat)
        return stats

    def evaluate_references(
        self,
        adapter_spec: AdapterSpec,
        reference_request_states: List[RequestState],
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """ Assign a score to the ranking of the references of an instance. """
        # Extract relevant data from the request states.
        ranking_objects: List[RankingObject] = [self.create_ranking_object(rs) for rs in reference_request_states]

        # Populate the model relevance of the ranking objects.
        ranking_objects = self.model_relevance_fn(ranking_objects)

        # Get ground truth relevances.
        true_relevances: Dict[int, int] = self.get_true_relevances(ranking_objects)

        # Get relevance values for each reference based on the model output and compute measures.
        run_relevances: Dict[int, int] = self.get_run_relevances(ranking_objects)
        stats: List[Stat] = self.compute_measures(true_relevances, run_relevances)

        # Get relevance values for each reference with rank up to self.rank based on model output and compute measures.
        if self.rank:
            run_relevances = self.get_run_relevances(ranking_objects, rank_limit=self.rank)
            mn_suffix = f" (topk={self.rank})"
            stats_with_rank_limit = self.compute_measures(true_relevances, run_relevances, metric_name_suffix=mn_suffix)
            stats += stats_with_rank_limit

        return stats
