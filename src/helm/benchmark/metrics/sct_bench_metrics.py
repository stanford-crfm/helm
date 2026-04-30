# The following scoring functions are adapted from:
# https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/modeling.py
#
# MIT License
#
# Copyright (c) 2025 liamgmccoy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


RATING_PATTERN = re.compile(r"\+2|\+1|0|-1|-2")


def parse_sct_rating(response: str) -> Optional[int]:
    # Adapted from:
    # https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/modeling.py#L190-L199
    try:
        rating_text = response.split("Rating: ")[1][:5]
        matches = RATING_PATTERN.findall(rating_text)
        label = int("".join(matches))
        assert label in [-2, -1, 0, 1, 2]
        return label
    except Exception:
        return None


def compute_sct_score(expert_distribution: List[float], rating: int) -> float:
    # Adapted from:
    # https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/modeling.py#L161-L169
    max_weight = max(expert_distribution)
    if max_weight == 0:
        return 0.0
    normalized = [w / max_weight for w in expert_distribution]
    rating_index = rating + 2
    return normalized[rating_index]


def compute_expert_set_membership(expert_distribution: List[float], rating: int) -> float:
    # Adapted from:
    # https://github.com/SCT-Bench/sctpublic/blob/c2843137b36218eac3f47e76eaa559c9fe7973e9/modeling.py#L171-L179
    rating_index = rating + 2
    return 1.0 if expert_distribution[rating_index] > 0 else 0.0


class SCTBenchMetric(Metric):
    """Metric for evaluating Script Concordance Test (SCT) responses.

    Computes two metrics:
    - sct_score: Normalized expert panel weight for the model's chosen rating (0 to 1).
    - sct_expert_set_membership: Whether the model's response falls within the expert
      agreement region (binary, averaged across instances).
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result, "request_state.result is unexpectedly None"
        predictions = [completion.text.strip() for completion in request_state.result.completions]

        if not predictions:
            hlog("Warning: No predictions found in completions")
            return []

        response_text = predictions[0]

        assert request_state.instance.extra_data, "extra_data with expert_distribution is required"
        expert_distribution = request_state.instance.extra_data["expert_distribution"]

        rating = parse_sct_rating(response_text)
        if rating is None:
            hlog(f"Warning: Could not parse SCT rating from response: {response_text[:100]}")
            return [
                Stat(MetricName("sct_score")).add(0.0),
                Stat(MetricName("sct_expert_set_membership")).add(0.0),
            ]

        sct_score = compute_sct_score(expert_distribution, rating)
        expert_membership = compute_expert_set_membership(expert_distribution, rating)

        return [
            Stat(MetricName("sct_score")).add(sct_score),
            Stat(MetricName("sct_expert_set_membership")).add(expert_membership),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="sct_score",
                display_name="SCT Score",
                short_display_name="SCT Score",
                description=(
                    "Average normalized expert panel weight for the model's chosen rating. "
                    "Higher means better agreement with clinical experts."
                ),
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="sct_expert_set_membership",
                display_name="Expert Set Membership",
                short_display_name="Expert Set %",
                description=(
                    "Percentage of model responses that fall within the expert agreement region "
                    "(non-zero expert weight)."
                ),
                lower_is_better=False,
                group=None,
            ),
        ]
