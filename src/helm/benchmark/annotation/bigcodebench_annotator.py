
import ast
import traceback
import time
import json

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.request import Request
from helm.common.hierarchical_logger import hlog

from typing import Any, List, Dict
from gradio_client import Client, handle_file
from tempfile import TemporaryDirectory
from tenacity import retry, stop_after_attempt, wait_fixed


OUTPUT_FILENAME = "tmp_result.jsonl"


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


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

    def __init__(self):
        self.remote_execute_api = "https://bigcode-bigcodebench-evaluator.hf.space/"
        self.split = "instruct"
        self.subset = "full"
        self.pass_k = "1"  # Original: "1,5,10"
        self.use_global_metric = True

    def annotate(self, request_state: RequestState) -> Any:
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(4))
    def predict_with_retry(self, filename):
        client = Client(self.remote_execute_api)
        results, pass_at_k = client.predict(
            split=self.split,
            subset=self.subset,
            samples=handle_file(filename),
            pass_k=self.pass_k,
            api_name="/predict",
        )
        results, pass_at_one = pass_at_k["pass@1"]
        return results, pass_at_one


    def annotate_all(self, request_states: List[RequestState]) -> List[Dict[str, Any]]:
        assert all(request_state.result for request_state in request_states)
        assert all(len(request_state.result.completions) == 1 for request_state in request_states)
        assert all(request_state.instance.extra_data for request_state in request_states)

        with TemporaryDirectory() as tmpdir:
            with open(OUTPUT_FILENAME, "w") as file:
                res = []
                for i in range(1140):
                    init_line = f'{{"task_id": "BigCodeBench/{i}", "solution": ""}}\n'
                    res.append(init_line)
                for request_state in request_states:
                    line: str
                    model_output_text = request_state.result.completions[0].text
                    solution = code_extract(model_output_text)
                    escaped_solution = json.dumps(solution)[1:-1]
                    idx = int(request_state.instance.id.split("/")[-1])
                    res[idx] = json.dumps(
                        {"task_id": request_state.instance.id, "solution": escaped_solution}
                    ) + "\n"
                for line in res:
                    file.write(line)

        try:
            results, pass_at_one = self.predict_with_retry(OUTPUT_FILENAME)
        except Exception as e:
            hlog("Failed to complete the operation after 3 attempts.")
            pass_at_one = 0.0
            results = []
        if len(results):
            ret = [{"pass_at_one": results['eval'][state.instance.id][0]['status'] == 'pass'} for state in request_states]
        else:
            ret = [{"pass_at_one": False} for state in request_states]
        return ret
