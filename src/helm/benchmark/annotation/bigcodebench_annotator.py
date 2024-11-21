from typing import Any

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.request import Request
from gradio_client import Client, handle_file
from tempfile import TemporaryDirectory

import ast
import traceback
import time
import json


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
        self.remote_execute_api = "https://bigcode-bigcodebench-evaluator-2.hf.space/"
        self.split = "instruct"
        self.subset = "full"
        self.pass_k = "1"  # Original: "1,5,10"

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        assert request_state.instance.extra_data
        model_output_text = request_state.result.completions[0].text
        solution = code_extract(model_output_text)

        pass_at_one: float
        with TemporaryDirectory() as tmpdir:
            
            # dump result to a jsonl in tmpdir using json library             
            with open(f"{tmpdir}/result.jsonl", "w") as file:
                for i in range(1140):
                    line: str
                    if request_state.instance.extra_data["task_id"] == f"BigCodeBench/{i}":
                        escaped_solution = json.dumps(solution)[1:-1]
                        line = f'{{"task_id": "BigCodeBench/{i}", "solution": "{escaped_solution}"}}\n'
                    else:
                        line = f'{{"task_id": "BigCodeBench/{i}", "solution": ""}}\n'
                    file.write(line)

            # with open(f"node_modules/temp_result.jsonl", "w") as file:
            #     for i in range(1140):
            #         line: str
            #         if request_state.instance.extra_data["task_id"] == f"BigCodeBench/{i}":
            #             escaped_solution = json.dumps(solution)[1:-1]
            #             line = f'{{"task_id": "BigCodeBench/{i}", "solution": "{escaped_solution}"}}\n'
            #         else:
            #             line = f'{{"task_id": "BigCodeBench/{i}", "solution": ""}}\n'
            #         file.write(line)

            # # Following https://github.dev/bigcode-project/bigcodebench/blob/main/bigcodebench/evaluate.py
            # while True:
            #     try:
            #         client = Client(self.remote_execute_api)
            #         results, pass_at_k = client.predict(
            #             split=self.split,
            #             subset=self.subset,
            #             samples=handle_file(f"{tmpdir}/result.jsonl"),
            #             pass_k=self.pass_k,
            #             api_name="/predict"
            #         )
            #         break
            #     except Exception as e:
            #         print(f"Error Message: {e}. Retrying in 4s...")
            #         time.sleep(4)

            max_retries = 3
            retry_count = 0
            success = False  # Flag to indicate if the operation was successful

            while retry_count < max_retries:
                try:
                    client = Client(self.remote_execute_api)
                    results, pass_at_k = client.predict(
                        split=self.split,
                        subset=self.subset,
                        samples=handle_file(f"{tmpdir}/result.jsonl"),
                        pass_k=self.pass_k,
                        api_name="/predict"
                    )
                    success = True  # Operation succeeded
                    pass_at_one = pass_at_k["pass@1"]
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Attempt {retry_count} failed. Error Message: {e}. Retrying in 4s...")
                    time.sleep(4)

            if not success:
                print("Failed to complete the operation after 3 attempts.")
                pass_at_one = 0


        return {"pass_at_one": pass_at_one}
