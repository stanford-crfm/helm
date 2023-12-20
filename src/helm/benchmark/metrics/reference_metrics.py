from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List
from helm.benchmark.metrics.metric_name import MetricName

from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import Instance, Reference


class ReferenceMetric(ABC):
    name: MetricName

    @abstractmethod
    def score(self, gold: str, pred: str) -> float:
        pass

    def compute_metric(self, instance: Instance, preds: List[str], top_k: int = 1) -> List[Stat]:
        golds: List[Reference] = [reference for reference in instance.references if reference.is_correct]
        assert len(golds) > 0

        score_1 = max(self.score(gold.output.text, preds[0]) for gold in golds)

        metrics = [Stat(self.name).add(score_1)]  # score_1 corresponds using one prediction
        if top_k != 1:
            score_k = max(self.score(gold.output.text, pred) for gold in golds for pred in preds)
            metrics.append(Stat(replace(self.name, name=f"{self.name.name}@{top_k}")).add(score_k))

        return metrics
