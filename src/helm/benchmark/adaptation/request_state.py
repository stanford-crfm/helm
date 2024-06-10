from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from helm.benchmark.scenarios.scenario import Instance
from helm.common.general import indent_lines, format_text_lines, serialize
from helm.common.request import Request, RequestResult


@dataclass(frozen=True)
class RequestState:
    """
    A `RequestState` represents a single `Request` made on behalf of an `Instance`.
    It should have all the information that's needed later for a `Metric` to be
    able to understand the `Request` and its `RequestResult`.
    """

    instance: Instance
    """Which instance we're evaluating"""

    reference_index: Optional[int]
    """Which reference of the instance we're evaluating (if any)"""

    request_mode: Optional[str]
    """Which request mode ("original" or "calibration") of the instance we're evaluating (if any)
    (for ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED)"""

    train_trial_index: int
    """Which training set this request is for"""

    output_mapping: Optional[Dict[str, str]]
    """How to map the completion text back to a real output (e.g., for multiple choice, "B" => "the second choice")"""

    request: Request
    """The request that is actually made"""

    result: Optional[RequestResult]
    """The result of the request (filled in when the request is executed)"""

    num_train_instances: int
    """Number of training instances (i.e., in-context examples)"""

    prompt_truncated: bool
    """Whether the prompt (instructions + test input) is truncated to fit the model's context window."""

    num_conditioning_tokens: int = 0
    """The number of initial tokens that will be ignored when computing language modeling metrics"""

    annotations: Optional[Dict[str, Any]] = None
    """Output of some post-processing step that is needed for the metric to understand the request
    Should match the annotator's name to an Annotation (usually a list of dictionaries for each completion)
    Example: parsing, rendering an image based on the text completion, etc."""

    def __post_init__(self):
        if self.request_mode:
            assert self.request_mode in ["original", "calibration"], f"Invalid request_mode: {self.request_mode}"

    def render_lines(self) -> List[str]:
        output = [f"train_trial_index: {self.train_trial_index}"]
        if self.reference_index:
            output.append(f"reference_index: {self.reference_index}")

        output.append("instance {")
        output.extend(indent_lines(self.instance.render_lines()))
        output.append("}")

        # Part of request but render multiline
        output.append("request.prompt {")
        output.extend(indent_lines(format_text_lines(self.request.prompt)))
        output.append("}")

        output.append("request {")
        output.extend(indent_lines(serialize(self.request)))
        output.append("}")

        if self.result:
            output.append("result {")
            output.extend(indent_lines(self.result.render_lines()))
            output.append("}")

        return output
