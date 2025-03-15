import re

from datetime import datetime
from typing import List, Dict, Any
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class MedCalcBenchMetric(Metric):
    """
    Metric for evaluating the MedCalc Bench dataset, assessing the model's ability to
    be a clinical calculator.

    Exact match based on category:
    1. Normal exact match: for categories "risk", "severity" or "diagnosis".
    2. Variant exact match: for other categories, if the number calculated by the model falls between the values
        in the Lower limit and Upper limit columns, we mark it as accurate.
    """

    def parse_duration(self, duration_str) -> int:
        """Parses a duration tuple (weeks, days) from a string format like ('14 weeks', '2 days')."""
        match = re.match(r"\('(\d+) weeks', '(\d+) days'\)", duration_str)
        if match:
            weeks, days = map(int, match.groups())
            return weeks * 7 + days  # Convert to total days
        else:
            raise ValueError(f"Invalid format: {duration_str}")

    def is_within_range(self, lower_bound, upper_bound, prediction) -> int:
        """
        Checks if a predicted duration falls within the given range.

        Args:
            lower_bound (str): The lower bound in format "('X weeks', 'Y days')".
            upper_bound (str): The upper bound in format "('X weeks', 'Y days')".
            prediction (str): The predicted duration in the same format.

        Returns:
            int: 1 if within range (inclusive), 0 otherwise.
        """
        lower_days = self.parse_duration(lower_bound)
        upper_days = self.parse_duration(upper_bound)
        prediction_days = self.parse_duration(prediction)
        return 1 if lower_days <= prediction_days <= upper_days else 0

    def check_date(self, prediction: str, reference: str, extra_data: Dict[str, Any]) -> int:
        """Checks if prediction date is withing limits"""
        if re.match(r"\('(\d+) weeks', '(\d+) days'\)", reference):
            exact_match = self.is_within_range(extra_data["lower_limit"], extra_data["upper_limit"], prediction)
        else:
            prediction_date = self._str_to_date(prediction)
            upper_limit_date = self._str_to_date(extra_data["upper_limit"])
            lower_limit_date = self._str_to_date(extra_data["lower_limit"])
            exact_match = 1 if lower_limit_date <= prediction_date <= upper_limit_date else 0
        return exact_match

    def _str_to_date(self, date_str: str) -> datetime:
        """Convert string to datetime object."""
        return datetime.strptime(date_str, "%m/%d/%Y")

    def check_in_range(self, prediction: str, reference: str, extra_data: Dict[str, Any], category: str) -> int:
        """Check if the prediction falls within the range specified by the reference."""
        try:
            if category == "date":
                exact_match = self.check_date(prediction, reference, extra_data)
            elif category in ["dosage conversion", "physical"]:
                lower_limit = float(extra_data["lower_limit"])
                upper_limit = float(extra_data["upper_limit"])
                float_prediction = float(prediction)
                exact_match = 1 if lower_limit <= float_prediction <= upper_limit else 0
            else:
                raise ValueError(f"Category {category} not supported")
        except ValueError:
            return 0

        return exact_match

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate a single generation against reference labels.
        """
        # Extract predictions
        assert request_state.result, "request_state.result is unexpectedly None"
        predictions = [completion.text.strip() for completion in request_state.result.completions]

        if not predictions:
            hlog("Warning: No predictions found in completions")
            return []

        # Get the first prediction
        prediction = predictions[0]

        # Get references
        references = getattr(request_state.instance, "references", None)

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        reference = references[0].output.text

        # Extract category, upper limit and lower limit
        assert request_state.instance.extra_data, "Extra data dict was expected but got None"
        category = request_state.instance.extra_data["category"]

        if category in ["risk", "severity", "diagnosis"]:
            exact_match = 1 if prediction == reference else 0
        else:
            exact_match = self.check_in_range(prediction, reference, request_state.instance.extra_data, category)

        return [
            Stat(MetricName("medcalc_bench_accuracy")).add(exact_match),
        ]
