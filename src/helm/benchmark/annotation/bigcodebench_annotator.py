from typing import Any, List, Dict, Optional
import ast
import json
import tempfile
import traceback

from gradio_client import Client, handle_file
from retrying import retry

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.hierarchical_logger import hlog


# Based on https://github.com/bigcode-project/bigcodebench/blob/0331489b29cbf2653b4669597ef431e158882aab/bigcodebench/syncheck.py#L14  # noqa: E501
# Licensed under Apache 2.0
def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


# Based on https://github.com/bigcode-project/bigcodebench/blob/0331489b29cbf2653b4669597ef431e158882aab/bigcodebench/sanitize.py#L30  # noqa: E501
# Licensed under Apache 2.0
def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


class BigCodeBenchAnnotator(Annotator):
    """The BigCodeBench autograder."""

    name = "bigcodebench"

    DEFAULT_URL = "https://bigcode-bigcodebench-evaluator.hf.space/"
    SPLIT = "instruct"
    SUBSET = "full"
    PASS_K = "1"
    DATASET_SIZE = 1140

    def __init__(self, api_key: Optional[str], endpoint: Optional[str]):
        self.use_global_metric = True
        if api_key and endpoint:
            hlog(f"BigCodeBenchAnnotator will use the configured endpoint {endpoint}")
            self.client = Client(endpoint, hf_token=api_key)
        else:
            hlog(
                f"WARNING: BigCodeBenchAnnotator will use the default public evaluator endpoint {self.DEFAULT_URL} - "
                "set bigcodebenchApiKey and bigcodebenchEndpoint in credentials.conf to use a cloned evaluator instead"
            )
            self.client = Client(self.DEFAULT_URL)

    def annotate(self, request_state: RequestState) -> Any:
        raise NotImplementedError("annotate() is not supported; use annotate_all() instead")

    @retry(stop_max_attempt_number=3, wait_fixed=4000)
    def send_request_to_gradio_evaluator(self, filename: str, task_ids: List[str]):
        if len(task_ids) == self.DATASET_SIZE:
            selective_evaluate = ""
        else:
            selective_evaluate = ",".join([task_id.removeprefix("BigCodeBench/") for task_id in task_ids])
        return self.client.predict(
            split=self.SPLIT,
            subset=self.SUBSET,
            samples=handle_file(filename),
            pass_k=self.PASS_K,
            api_name="/predict",
            selective_evaluate=selective_evaluate,
        )

    def annotate_all(self, request_states: List[RequestState]) -> List[Dict[str, Any]]:
        task_id_to_solution: Dict[str, str] = {}
        for request_state in request_states:
            assert request_state.instance.id is not None
            task_id = request_state.instance.id
            assert request_state.result is not None
            assert len(request_state.result.completions) == 1
            model_output_text = request_state.result.completions[0].text
            solution = code_extract(model_output_text)
            task_id_to_solution[task_id] = solution

        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            hlog(f"Temporary file for BigCodeBenchAnnotator: {temp_file.name}")
            with open(temp_file.name, "w") as f:
                for task_id, solution in task_id_to_solution.items():
                    json.dump({"task_id": task_id, "solution": solution}, f)
                    f.write("\n")
            eval_result = self.send_request_to_gradio_evaluator(temp_file.name, list(task_id_to_solution.keys()))[0]
        return [
            {"bigcodebench": {"pass_at_one": eval_result["eval"][request_state.instance.id][0]["status"] == "pass"}}
            for request_state in request_states
        ]
