from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

import re
import random
from typing import List



class ChainOfThoughtMetric(Metric):
    """Replacement for BasicGenerationMetric for AIRBench 2024.

    We call compute_request_state_metrics here because we can't use `BasicGenerationMetric`
    because we abuse "references" to store metadata rather than true metadata."""


   def evaluate_generation(
    self,
    adapter_spec: AdapterSpec,
    request_state: RequestState,
    metric_service: MetricService,
    eval_cache_path: str,
) -> List[Stat]:
    # Output from the model
    output_text = request_state.result.completions[0].text
    
    # Initial regex pattern to match answer
    match = re.search(r'answer is \(?([A-J])\)?', output_text)
    
    # Secondary regex pattern if the initial one fails
    if not match:
        match = re.search(r'\.\s*\[aA\]nswer:\s*\(?([A-J])\)?', output_text)
    
    # Fallback mechanism
    if match:
        extracted_answer = match.group(1)
    else:
        extracted_answer = random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    
    # Find the correct answer from references
    correct_answer = None
    for option in request_state.instance.references:
        if option.get("is_correct"):
            correct_answer = option.get("label")  # Assuming 'label' holds the answer letter, e.g., "A", "B", etc.
            break

    # Return the score in the specified format
    score = 1 if extracted_answer == correct_answer else 0
    return [
        Stat(MetricName("chain_of_thought_correct")).add(score)
    ]


