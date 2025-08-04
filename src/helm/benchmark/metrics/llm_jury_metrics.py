from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.common.hierarchical_logger import hlog, hwarn
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


@dataclass
class RubricItem:
    name: str
    min: float
    max: float
    weight: float
    higher_is_better: bool


@dataclass
class Rubric:
    items: Dict[str, RubricItem]

    @classmethod
    def from_config(cls, rubric_config: Dict[str, Any]) -> "Rubric":
        items = {}
        for name, attrs in rubric_config.items():
            item = RubricItem(
                name=name,
                min=attrs["min"],
                max=attrs["max"],
                weight=attrs["weight"],
                higher_is_better=attrs["higher_is_better"],
            )
            items[name] = item
        return cls(items)

    def normalize(self, name: str, score: float) -> float:
        """Normalize the score according to the rubric item config."""
        item = self.items[name]
        raw = (score - item.min) / (item.max - item.min)
        return raw if item.higher_is_better else 1 - raw

    def aggregate(self, scores: Dict[str, float]) -> float:
        """Weighted aggregation of normalized scores."""
        total = 0.0
        weight_offset = 0.0
        invalid_scores = [name for name in scores.keys() if not isinstance(scores[name], (int, float))]
        if invalid_scores:
            n_valid_scores = len(scores) - len(invalid_scores)
            weight_offset = sum(self.items[name].weight for name in invalid_scores) / n_valid_scores
            hwarn(
                f"Invalid scores found for {invalid_scores}. "
                f"Using average weight offset of {weight_offset} to adjust the total score."
            )
        for name, score in scores.items():
            if not isinstance(score, (int, float)):
                hwarn(f"Skipping non-numeric score for {name}: {score}")
                continue
            norm = self.normalize(name, score)
            total += norm * (self.items[name].weight + weight_offset)
        return total


class LLMJuryMetric(Metric):
    """Score metrics for LLM Jury."""

    def __init__(
        self,
        metric_name: str,
        scenario_name: str,
        annotator_models: Dict[str, AnnotatorModelInfo],
        default_score: float = 0.0,
        rubric: Optional[Rubric] = None,
    ):
        self.metric_name = metric_name
        self.scenario_name = scenario_name
        self.annotator_models = annotator_models
        self.default_score = default_score
        self.rubric = rubric

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        if self.rubric:
            hlog(f"Using rubric for {self.scenario_name} with items: {list(self.rubric.items.keys())}")
        else:
            hlog(f"No rubric defined for {self.scenario_name}, using raw scores.")
        annotations: Dict[str, Any] = request_state.annotations[self.scenario_name]
        scores: List[int] = []
        score = self.default_score
        for annotation_key, annotation_dict in annotations.items():
            if annotation_key in self.annotator_models.keys() and annotation_dict is not None:
                if self.rubric:
                    # Use rubric to normalize and aggregate scores
                    scores_dict = {
                        item: annotation_dict[item]["score"]
                        for item in self.rubric.items.keys()
                        if item in annotation_dict
                    }
                    score = self.rubric.aggregate(scores_dict)
                else:
                    # Fallback to using the raw score
                    for val in annotation_dict.values():
                        scores.append(int(val["score"]))
                    if scores:
                        score = sum(scores) / len(scores)
        return [
            Stat(MetricName(self.metric_name)).add(score),
        ]
