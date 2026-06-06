from typing import List, Dict, Any, cast

import pandas as pd

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class EclekticMetric(EvaluateInstancesMetric):
    """Score metrics for Eclektic."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:

        assert request_state.annotations is not None
        scores = request_state.annotations["eclektic_autograder"]

        return [Stat(MetricName("accuracy")).add(scores["correct"])]

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        # ----------------------------------------------------------
        # Record the fields we need for the corpus‑level calculations
        # ----------------------------------------------------------
        data_rows: List[Dict[str, object]] = []
        for req_state in request_states:
            if req_state is None:
                raise ValueError("RequestState does not exist")

            # Ensure annotations exist and have the expected key
            if req_state.annotations is None:
                raise ValueError("Annotations not found")
            if "eclektic_autograder" not in req_state.annotations:
                raise ValueError("Annotation not found")

            ann: Dict[str, Any] = cast(Dict[str, Any], req_state.annotations["eclektic_autograder"])

            # Handle optional extra_data safely
            extra_data: Dict[str, Any] = req_state.instance.extra_data or {}

            data_rows.append(
                {
                    "instance_id": req_state.instance.id,
                    "lang": extra_data.get("lang"),
                    "original_lang": extra_data.get("original_lang"),
                    "correct": bool(ann.get("correct", False)),
                }
            )

        if data_rows:  # Skip if evaluation somehow produced no data
            data = pd.DataFrame(data_rows)

            # Questions answered correctly in their *original* language
            correct_in_lang_qids = set(
                data[(data["correct"]) & (data["lang"] == data["original_lang"])]["instance_id"].tolist()
            )

            # ------------------ overall (translated only) ------------------
            scored_data = data[data["lang"] != data["original_lang"]]
            if not scored_data.empty:
                overall_successes = scored_data[
                    (scored_data["correct"]) & (scored_data["instance_id"].isin(correct_in_lang_qids))
                ]
                overall_score = len(overall_successes) / len(scored_data)
            else:
                overall_score = 0.0

            # ------------- overall_transfer (all languages) ---------------
            transfer_data = data[data["instance_id"].isin(correct_in_lang_qids)]
            if not transfer_data.empty:
                transfer_successes = transfer_data[
                    (transfer_data["correct"]) & (transfer_data["instance_id"].isin(correct_in_lang_qids))
                ]
                transfer_score = len(transfer_successes) / len(transfer_data)
            else:
                transfer_score = 0.0

            return [
                Stat(MetricName("overall")).add(overall_score),
                Stat(MetricName("overall_transfer")).add(transfer_score),
            ]

        return [
            Stat(MetricName("overall")).add(0.0),
            Stat(MetricName("overall_transfer")).add(0.0),
        ]
