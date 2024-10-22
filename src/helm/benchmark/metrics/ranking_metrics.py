from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_RANKING_BINARY
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.reference_metric import ReferenceMetric
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.scenarios.scenario import unpack_tag, CORRECT_TAG, Reference
from helm.common.request import RequestResult
from helm.common.general import assert_present, binarize_dict
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

try:
    import pytrec_eval
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["metrics"])


@dataclass
class RankingObject:
    """Simple dataclass used to store features used for ranking objects."""

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

    """ Model output for the object.

    Model output can be any string outputted by the model that's used to rank
    a model's preference on different passages. For example, it can be the
    string "Yes" or "No" indicating whether a passage answers the given query
    in the ADAPT_RANKING_BINARY case.
    """
    model_output: Optional[str] = None

    """ Model logprob for the completion. """
    model_logprob: Optional[float] = None

    """ Model relevance.

    Relevance of this object for the query as determined by the model output
    and logprob.
    """
    model_relevance: Optional[int] = None


class RankingMetric(ReferenceMetric):
    """Ranking metric."""

    """ Methods supported by this metric.

    Following methods are supported by this metric:
        (1) ADAPT_RANKING_BINARY: In binary_ranking method, the model's task is
            to predict whether a given context contains an answer to a given
            query. Different contexts for a query are then sorted using the
            label outputted by model as well as the model's confidence in the
            label.
    """
    METHOD_LIST = [ADAPT_RANKING_BINARY]

    """ Measures. """
    RECIP_RANK_MEASURE = "recip_rank"
    SUCCESS_MEASURE = "success"
    RECALL_MEASURE = "recall"
    NDCG_CUT_MEASURE = "ndcg_cut"
    SUPPORTED_MEASURES = [RECIP_RANK_MEASURE, SUCCESS_MEASURE, RECALL_MEASURE, NDCG_CUT_MEASURE]
    BINARY_MEASURES = [RECIP_RANK_MEASURE, RECALL_MEASURE, SUCCESS_MEASURE]
    CUSTOM_PARAMETRIZED_MEASURES = [RECIP_RANK_MEASURE]

    """ Token used to represent missing results.

    Results may be missing if the RequestState object didn't complete.
    """
    MISSING_RESULT_TEXT = "missing"

    def __init__(
        self,
        method: str,
        measure_names: List[str],
        correct_output: str,
        wrong_output: str,
        rank: Optional[int] = None,
        multiple_relevance_values: bool = False,
    ):
        """Constructor for the RankingMetric.

        Args:
            method: The adaptation method used. The method must exists in
                self.METHOD_LIST.
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
                evaluation. If None, all the rankings are evaluated. If
                specified, only the documents that have a rank up to and
                including the specified rank are evaluated.
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
        assert method in self.METHOD_LIST, f"Mode must be one of {self.METHOD_LIST}, instead found {method}."
        self.method = method

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
        METHOD_TO_RELEVANCE_FUNCTION: Dict[str, Callable[[List[RankingObject]], List[RankingObject]]] = {
            ADAPT_RANKING_BINARY: self.compute_model_relevance_binary_ranking
        }
        self.model_relevance_fn = METHOD_TO_RELEVANCE_FUNCTION[self.method]

    @staticmethod
    def parse_measure_name(measure_name: str) -> Tuple[str, Optional[int]]:
        """Parse measure name of the form 'measure_name' or 'measure_name.k' into its parts."""
        measure_delimiter = "."
        if measure_delimiter in measure_name:
            measure_name, k = measure_name.split(measure_delimiter)
            return measure_name, int(k)
        return measure_name, None

    def validate_measure_name(self, measure_name: str) -> bool:
        """Check if the provided measure_name is a valid measure name."""
        measure_name, k = self.parse_measure_name(measure_name)
        if measure_name in self.SUPPORTED_MEASURES + list(pytrec_eval.supported_measures):
            return True
        return False

    def measure_to_metric_name(self, measure_name: str) -> str:
        """Convert the measure name to an easier to read metric name."""
        MEASURE_TO_METRIC_NAME = {"ndcg_cut": "NDCG", "recip_rank": "RR"}
        parsed_measure_name, k = self.parse_measure_name(measure_name)
        metric_name_str = MEASURE_TO_METRIC_NAME.get(parsed_measure_name, parsed_measure_name.capitalize())
        k_str = f"@{k}" if k else ""
        return f"{metric_name_str}{k_str}"

    @staticmethod
    def standardize(text: str) -> str:
        """Strip and lower the given text."""
        return text.strip().lower()

    @staticmethod
    def get_tag_value(tag) -> str:
        _, value = unpack_tag(tag)
        return value

    def get_query_string(self, qid):
        return f"q{qid}"

    def get_run_relevances(self, ranking_objs: List[RankingObject], rank_limit: Optional[int] = None) -> Dict[str, int]:
        """Get the relevance dictionary for the run.

        The run relevance dictionary contains the relevance values of each
        document as determined by the model. It is compared to the true
        relevance dictionary, which contains the ground truth relevance
        values for each document.
        """
        if rank_limit:
            return {
                self.get_query_string(r.reference_index): assert_present(r.model_relevance)
                for r in ranking_objs
                if r.rank and r.rank <= rank_limit
            }
        return {self.get_query_string(r.reference_index): assert_present(r.model_relevance) for r in ranking_objs}

    def get_true_relevances(self, ranking_objects: List[RankingObject]) -> Dict[str, int]:
        """Get the true relevance dictionary."""
        return {self.get_query_string(obj.reference_index): obj.relevance for obj in ranking_objects if obj.relevance}

    def object_to_score_binary_ranking(self, ranking_object: RankingObject) -> Tuple[int, float]:
        """Score the given ranking object."""
        # Input validation
        assert ranking_object.model_output is not None
        model_output = self.standardize(ranking_object.model_output)
        assert ranking_object.model_logprob is not None
        logprob = ranking_object.model_logprob

        # Score
        if model_output == self.standardize(self.correct_output):
            return 2, logprob
        elif model_output == self.standardize(self.wrong_output):
            return 1, -1 * logprob
        else:
            return 0, -1 * logprob

    def compute_model_relevance_binary_ranking(self, ranking_objects: List[RankingObject]) -> List[RankingObject]:
        """Populate the model_relevance field for the ranking objects."""
        # Sort the ranking objects
        ranking_objects_sorted = sorted(ranking_objects, key=self.object_to_score_binary_ranking, reverse=True)

        # Add model relevance
        for ind, ranking_object in enumerate(ranking_objects_sorted):
            ranking_object.model_relevance = len(ranking_objects_sorted) - ind

        return ranking_objects_sorted

    def get_model_output_logprob(self, result: RequestResult) -> Tuple[str, float]:
        """Get model's completion and the relevant logprob from a request state."""
        model_output, model_logprob = self.MISSING_RESULT_TEXT, 0.0
        if result and result.completions:
            # TODO: Decide how to pick between multiple completions.
            model_output = result.completions[0].text  # Model completion, which might contain whitespace
            model_logprob = result.completions[0].logprob
        return model_output, model_logprob

    def create_ranking_object(self, request_state: RequestState) -> RankingObject:
        """Create a RankingObject from a RequestState."""
        # Get reference index
        assert request_state.reference_index is not None
        reference_index: int = request_state.reference_index
        reference: Reference = request_state.instance.references[request_state.reference_index]

        # Get gold, relevance and rank fields
        is_gold: bool = CORRECT_TAG in reference.tags
        relevance, rank = None, None
        for tag in reference.tags:
            if "relevance" in tag:
                relevance = int(self.get_tag_value(tag))
            elif "rank" in tag:
                rank = int(self.get_tag_value(tag))

        # Get model completion and the associated log probability
        assert request_state.result is not None
        model_output, model_logprob = self.get_model_output_logprob(request_state.result)

        # Create a RankingObject
        ranking_object = RankingObject(
            reference_index=reference_index,
            text=reference.output.text,
            gold=is_gold,
            relevance=relevance,
            rank=rank,
            model_output=model_output,
            model_logprob=model_logprob,
        )
        return ranking_object

    @staticmethod
    def limit_relevance_dict(relevance_dict: Dict[str, int], k: int) -> Dict[str, int]:
        """Return a dictionary containing the k pairs with the highest relevance values."""
        # Sort the dictionary by the relevance values and limit the number of elements.
        relevance_dict = {k: v for k, v in sorted(relevance_dict.items(), key=lambda item: item[1], reverse=True)}
        limit = min(len(relevance_dict), k)  # Limit the dictionary
        relevance_dict = {key: relevance_dict[key] for key in list(relevance_dict.keys())[:limit]}
        return relevance_dict

    def compute_measure(
        self, true_relevances: Dict[str, int], run_relevances: Dict[str, int], evaluator_measure_name: str
    ) -> float:
        """Compute measure_name on the true and run relevance values provided.

        Return a dictionary mapping metric names to the computed values.
        """
        # Parse the measure name
        parsed_measure_name, k = self.parse_measure_name(evaluator_measure_name)

        # If the measure is a binary measure, we binarize the true relevance values.
        if parsed_measure_name in self.BINARY_MEASURES and self.multiple_relevance_values:
            true_relevances = binarize_dict(true_relevances)

        # pytrec doesn't support parametrization for some measure names. For
        # such measures, we achieve parametrization by simply limiting
        # run_relevances dictionary to the top k relevance values.
        if parsed_measure_name in self.CUSTOM_PARAMETRIZED_MEASURES and k:
            evaluator_measure_name = parsed_measure_name  # Pass the non-parametrized name to pytrec
            run_relevances = self.limit_relevance_dict(run_relevances, k)

        # RelevanceEvaluator expects a dictionary that maps Query IDs to their
        # relevance dictionaries. To pass our relevance dictionaries in the
        # right format, we wrap them in a dictionary.
        measure_names, true_qrels, run_qrels = {evaluator_measure_name}, {"_": true_relevances}, {"_": run_relevances}
        evaluator = pytrec_eval.RelevanceEvaluator(true_qrels, measure_names)
        evaluations = evaluator.evaluate(run_qrels)
        score = list(evaluations["_"].values())[0]
        return score

    def compute_measures(
        self, true_relevances: Dict[str, int], run_relevances: Dict[str, int], metric_name_suffix=""
    ) -> List[Stat]:
        """Compute self.measures on the true and run relevances provided."""
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
        """Assign a score to the ranking of the references of an instance."""
        # Extract relevant data from the request states.
        ranking_objects: List[RankingObject] = [self.create_ranking_object(rs) for rs in reference_request_states]

        # Populate the model relevance of the ranking objects.
        ranking_objects = self.model_relevance_fn(ranking_objects)

        # Get ground truth relevances.
        true_relevances: Dict[str, int] = self.get_true_relevances(ranking_objects)

        # Get relevance values for each reference based on the model output and compute measures.
        run_relevances: Dict[str, int] = self.get_run_relevances(ranking_objects)
        stats: List[Stat] = self.compute_measures(true_relevances, run_relevances)

        # Get relevance values for each reference with rank up to self.rank based on model output and compute measures.
        if self.rank:
            run_relevances = self.get_run_relevances(ranking_objects, rank_limit=self.rank)
            mn_suffix = f" (topk={self.rank})"
            stats_with_rank_limit = self.compute_measures(true_relevances, run_relevances, metric_name_suffix=mn_suffix)
            stats += stats_with_rank_limit

        # Add reference ranks as stats
        # - Ranking objects with higher model relevance should have a better
        #   ranking, which is why we are setting the ranking of an object to be
        #   len(ranking_objects) minus its relevance.
        stats += [
            Stat(MetricName(f"ref{r.reference_index}_rank")).add(
                len(ranking_objects) - assert_present(r.model_relevance)
            )
            for r in ranking_objects
        ]

        return stats
