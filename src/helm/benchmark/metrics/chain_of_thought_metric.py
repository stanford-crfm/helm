from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

import re


def extract_answer(output_text: str) -> str:
    """
    Extracts the answer from the output text using two exact regex patterns.
    Returns "N/A" if no valid answer is found.
    """
    # First regex: Matches "answer is (A-J)" with optional parentheses
    match = re.search(r"answer is \(?([A-J])\)?", output_text)
    if match:
        return match.group(1)

    # Second regex: Matches "[answer: (A-J)]" with optional leading characters like "."
    match = re.search(r"\.*\[aA\]nswer:\s*\(?([A-J])\)?", output_text)
    if match:
        return match.group(1)

    # If neither regex matches, return "N/A"
    return "N/A"


class ChainOfThoughtMetric(Metric):
    """Replacement for BasicGenerationMetric for AIRBench 2024."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # Output from the model
        if request_state.result is not None and request_state.result.completions:
            output_text = request_state.result.completions[0].text

        # Extract the answer using the updated logic
        extracted_answer = extract_answer(output_text)

        # Find the correct answer from references by translating index to letter
        correct_answer = None
        for index, option in enumerate(request_state.instance.references):
            if option.is_correct:
                correct_answer = chr(65 + index)  # Translate index (0 -> A, 1 -> B, etc.)
                break

        print(request_state.instance.id, correct_answer, extracted_answer, extracted_answer == correct_answer)

        # Return the score in the specified format
        return [Stat(MetricName("chain_of_thought_correct")).add(extracted_answer == correct_answer)]
