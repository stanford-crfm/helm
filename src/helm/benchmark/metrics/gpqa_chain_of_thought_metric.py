import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def extract_answer(output_text: str) -> Optional[str]:
    """
    Extracts the answer from the output text using two exact regex patterns.
    Returns None if no valid answer is found.

    Args:
        output_text (str): The text from which to extract the answer.

    Returns:
        Optional[str]: The extracted answer (A-J) if found, otherwise None.
    """
    # First regex: Matches "answer is (A-J)" with optional parentheses
    match = re.search(r"answer is \(?([A-J])\)?", output_text)
    if match:
        return match.group(1)

    # Second regex: Matches "[answer: (A-J)]" with optional leading characters like "."
    match = re.search(r"\.*\[aA\]nswer:\s*\(?([A-J])\)?", output_text)
    if match:
        return match.group(1)

    # Third regex: Matches "answer is (A-J)" with optional leading non-alpha characters
    match = re.search(r"correct answer is [^A-Za-z]*([A-J])", output_text)
    if match:
        return match.group(1)

    # Fourth regex: Matches "answer is (A-J)" with optional leading non-capital alpha characters
    match = re.search(r"correct answer is [^A-Z]*([A-J])", output_text)
    if match:
        return match.group(1)

    # If no regex matches, return None
    return None


class GPQAChainOfThoughtMetric(Metric):
    """
    This metric focuses on structured reasoning and the accuracy of extracted answers.
    It compares model outputs against correct answers provided in a multiple-choice
    format and returns a score indicating the correctness of the generated response.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate the generated output for chain-of-thought reasoning accuracy.

        The method extracts the model's output, determines the correct answer
        from the provided references, and compares the two to compute a binary score.

        Args:
            adapter_spec (AdapterSpec): Specification of the adapter used for the evaluation.
            request_state (RequestState): The state of the current request, including
                                          the input instance, output results, and references.
            metric_service (MetricService): A service used to compute metrics if needed.
            eval_cache_path (str): Path to the evaluation cache for storing or retrieving data.

        Returns:
            List[Stat]: A list containing a single `Stat` object with the correctness
                        score (1 for correct, 0 for incorrect) under the metric
                        name "chain_of_thought_correct".
        """
        # Assert that completions exist if the result is not None
        assert (
            request_state.result is not None and request_state.result.completions
        ), "Request state result must have completions."

        # Set output_text if the assertion passes
        output_text = request_state.result.completions[0].text

        # Extract the answer using the updated logic
        extracted_answer = extract_answer(output_text)

        # Find the correct answer from references by translating index to letter
        correct_answer = None
        for index, option in enumerate(request_state.instance.references):
            if option.is_correct:
                correct_answer = chr(65 + index)  # Translate index (0 -> A, 1 -> B, etc.)
                break

        # Raise an exception if no correct answer is found
        if correct_answer is None:
            raise ValueError(f"No correct answer found for instance ID {request_state.instance.id}")

        # Compare extracted answer with the correct answer and compute the score
        score = 1 if extracted_answer == correct_answer else 0
        return [Stat(MetricName("chain_of_thought_correctness")).add(score)]
