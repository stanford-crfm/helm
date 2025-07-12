from typing import List
import re
import os
import subprocess
import tempfile
import shutil

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def compile_code(i, temp_dir, timeout=10):
    """
    Compiles the C++ file at temp_dir/tc_{i}.cpp and outputs to temp_dir/tc_{i}.out.

    Args:
        i (int): Index of the code to compile.
        temp_dir (str): Temporary directory where the C++ files are located.

    Returns:
        str or None: Path to the executable if compilation succeeds, else None.
    """
    executable = os.path.join(temp_dir, f"tc_{i}.out")
    cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")

    try:
        result = subprocess.run(
            ["timeout", str(timeout), "g++", "-std=c++11", cpp_file, "-o", executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Optional: to get output as string
            timeout=timeout + 2,  # Optional: timeout for compilation
        )
        if result.returncode != 0:
            # print(f"Compilation failed for {cpp_file}:\n{result.stderr}")
            return None
        return executable
    except Exception as e:
        print(f"An error occurred while compiling {cpp_file}: {e}")
        return None


def parallel_compile(codes, temp_dir, timeout=10, max_workers=4):
    """
    Compiles multiple C++ codes in parallel.

    Args:
        codes (list): List of code snippets or identifiers.
        temp_dir (str): Directory containing the C++ files.
        max_workers (int): Maximum number of worker processes.

    Returns:
        list: List of paths to the compiled executables or None for failed compilations.
    """
    executables = []
    for i in range(len(codes)):
        executable = compile_code(i, temp_dir, timeout)
        executables.append(executable)

    return executables


def run_executable(executable, std_in, timeout=10):
    """
    Runs an executable with a timeout and captures its output.

    Args:
        executable (str): Path to the executable to run.
        timeout (int): Timeout for running the executable in seconds.

    Returns:
        tuple: (return_code, output) where return_code is 0 if successful, non-zero otherwise,
               and output is the stdout captured from the execution.
    """
    if executable is None:
        return (0, "")  # Return 0 and empty output for failed compilations

    try:
        result = subprocess.run(
            ["timeout", str(timeout), executable],
            input=std_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # To decode stdout and stderr as strings
            timeout=timeout + 2,  # Add a small buffer to the timeout
        )
        return (result.returncode, result.stdout)
    except Exception as e:
        print(f"An error occurred while running {executable}: {e}")
        return (1, "")  # Non-zero return code for errors


def parallel_run_executables(executables, std_inputs, timeout=10, max_workers=4):
    """
    Runs multiple executables in parallel with a timeout.

    Args:
        executables (list): List of paths to the executables.
        max_workers (int): Maximum number of worker processes.

    Returns:
        list: List of results containing the outputs from running each executable.
    """
    results = []
    for std_in, executable in zip(std_inputs, executables):
        result_code, output = run_executable(executable, std_in, timeout)
        results.append((result_code, output))

    return results


class CPPEvaluator:
    def __init__(self, template, testcases, timeout=10, max_workers=8):
        """Initializes the CPPEvaluator class.

        Args:
            template (str): The template code with placeholders for the student's answer and test cases.
            testcases (Dict[str]): A list of test cases, each containing the input, output, and optional std_in.
            max_workers (int, optional): The maximum number of workers to use for parallel processing. Defaults to 8.
        """
        self.template = template
        self.testcases = testcases
        self.timeout = timeout
        self.max_workers = max_workers
        self.formatted_testcases, self.std_inputs = self.format_testcases()

    def format_testcases(self):
        """Formats the test cases into the required format for the grading engine.

        Returns:
            Tuple[List[Dict[str]], List[str]]: A tuple containing the formatted test cases and standard inputs.
        """
        formatted_testcases = []
        std_inputs = []
        for testcase in self.testcases:
            formatted_testcases.append(
                {
                    "extra": "",
                    "testcode": testcase["input"],
                    "expected_output": testcase["output"],
                }
            )
            if "std_in" not in testcase:
                std_inputs.append("")
            else:
                std_inputs.append(testcase["std_in"])
        return formatted_testcases, std_inputs

    def generate_code(self, student_answer):
        """Generates the C++ code with the student's answer and test cases.

        Args:
            student_answer (str): The student's answer to be inserted into the template.

        Returns:
            List[str]: A list of C++ code snippets with the student's answer and test cases inserted.
        """
        # Insert the student's answer and test cases into the template
        code = self.template.replace("{{ STUDENT_ANSWER }}", student_answer)

        # Find the for loop in the template
        start_index = code.find("{% for TEST in TESTCASES %}")
        end_index = code.find("{% endfor %}") + len("{% endfor %}")

        list_codes = []
        for testcase in self.formatted_testcases:
            # Insert the test case code into the template between the for loop
            testcode = code[:start_index] + testcase["testcode"] + code[end_index:]
            list_codes.append(testcode)

        return list_codes

    def write_and_compile_code(self, codes):
        """Writes and compiles the C++ code.

        Args:
            codes (List[str]): A list of C++ code snippets.

        Returns:
            Tuple[List[str], str]: A tuple containing the list of executable paths and the temporary directory.
        """
        # Write the C++ code to a temporary file
        temp_dir = tempfile.mkdtemp()
        for i, code in enumerate(codes):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            with open(cpp_file, "w") as file:
                file.write(code)

        # Compile the C++ code
        executables = parallel_compile(codes, temp_dir, timeout=self.timeout, max_workers=self.max_workers)

        return executables, temp_dir

    def evaluate(self, student_answer):
        """Evaluates the student's answer using the test cases.

        Args:
            student_answer (str): The student's answer to be evaluated.

        Returns:
            Dict[str, Union[float, List[int]]]: A dictionary containing the score and test case results.
        """
        # Generate the C++ code with the student's answer
        codes = self.generate_code(student_answer)

        # Write and compile the C++ code
        executables, temp_dir = self.write_and_compile_code(codes)
        list_result = []

        executation_results = parallel_run_executables(
            executables, self.std_inputs, timeout=self.timeout, max_workers=self.max_workers
        )
        for i, testcase in enumerate(self.testcases):
            if executation_results[i][0] != 0:
                list_result.append(0)
                continue

            expected_output = testcase["output"]
            student_output = executation_results[i][1]
            if expected_output.strip() != student_output.strip():
                list_result.append(0)
            else:
                list_result.append(1)

        # Delete the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if len(list_result) == 0:
            return {"score": 0, "testcases": list_result}

        return {
            "score": sum(list_result) / len(list_result),
            "testcases": list_result,
        }


class CodeInsightsFunctionalCorrectnessMetric(Metric):
    """
    Metric for evaluating functional correctness of C++ code generation.

    Measures each model's functional correctness by computing the proportion of problems
    for which its generated code passes all provided unit tests. For every generated solution,
    we compile the C++ code (using g++) and execute the full test cases. We record the
    proportions of the unit test that passes for each problem and then take the mean across
    all problems. This yields a score between 0 and 1, where 1 indicates the model produced
    flawless codes, and lower values reveal the fraction of tasks it could not solve all
    the unit test cases.
    """

    def __init__(self, timeout: int = 10, max_workers: int = 8):
        """
        Initializes the CodeInsightsFunctionalCorrectnessMetric.

        Args:
            timeout (int): Timeout for each test case execution.
            max_workers (int): Maximum number of workers for parallel processing.
        """
        super().__init__()
        self.timeout = timeout
        self.max_workers = max_workers

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate LLM-generated code by running unit tests and computing pass rate.

        Returns:
            List of Stat objects containing the functional correctness score
        """
        print("\n=== FUNCTIONAL CORRECTNESS METRIC DEBUG ===")
        print(f"Instance ID: {getattr(request_state.instance, 'id', 'UNKNOWN')}")

        # Get the generated code from the request state
        if not request_state.result or not request_state.result.completions:
            print("ERROR: No output generated")
            return self._create_failure_stats("No output generated")

        generated_code = request_state.result.completions[0].text.strip()
        generated_code = self._extract_student_code(generated_code)
        print(f"Generated code length: {len(generated_code)}")
        print(f"Generated code preview: {generated_code[:200]}...")

        # Get test cases from instance extra_data
        if not hasattr(request_state.instance, "extra_data") or not request_state.instance.extra_data:
            print("ERROR: No extra_data available")
            print(f"Instance attributes: {dir(request_state.instance)}")
            return self._create_failure_stats("No test data available")

        extra_data = request_state.instance.extra_data
        print(f"Extra data keys: {list(extra_data.keys())}")

        test_cases = extra_data.get("test_cases", [])
        question_template = extra_data.get("question_template", "")
        question_name = extra_data.get("question_name", "UNKNOWN")

        print(f"Question name: {question_name}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"Template length: {len(question_template)}")

        if not test_cases:
            print("ERROR: No test cases available")
            return self._create_failure_stats("No test cases available")

        print(f"First test case preview: {test_cases[0] if test_cases else 'NONE'}")

        # Run unit tests and calculate pass rate
        evaluator = CPPEvaluator(question_template, test_cases, timeout=self.timeout, max_workers=self.max_workers)
        pass_rate = evaluator.evaluate(generated_code)["score"]

        print(f"Final pass rate: {pass_rate}")
        print("=== END DEBUG ===\n")

        return [Stat(MetricName("functional_correctness")).add(pass_rate)]

    def _extract_student_code(self, model_code: str) -> str:
        """
        Extracts clean C++ code from model output:
        - Trims preambles
        - Removes student's main()
        """
        code_blocks = re.findall(r"```(?:c\+\+)?\n(.*?)```", model_code, flags=re.DOTALL)
        if code_blocks:
            model_code = code_blocks[0].strip()  # Use the first code block
            print("[Markdown extraction] Used fenced code blocks.")

        # Post-processing
        # Comment out as a testing - 7/3/2025
        lines = model_code.strip().splitlines()
        start_keywords = ("#include", "using namespace")
        for i, line in enumerate(lines):
            if any(line.strip().startswith(k) for k in start_keywords):
                lines[i] = ""
        code = "\n".join(lines).strip()
        if "int main" in code:
            code = code.split("int main")[0].strip()

        # --- Final touch ---
        if "print(" in code and "void print()" not in code and "print()" not in code:
            print("⚠️ WARNING: `print()` is called in test input but not defined.")

        return code

    def _create_failure_stats(self, error_message: str) -> List[Stat]:
        """
        Create default statistics for failure cases.

        Args:
            error_message: Description of the failure

        Returns:
            List containing a single Stat with 0.0 functional correctness score
        """
        print(f"METRIC FAILURE: {error_message}")
        return [Stat(MetricName("functional_correctness")).add(0.0)]
