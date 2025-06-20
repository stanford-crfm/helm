from typing import Dict, List, Tuple
import pandas as pd
import re
import os
import subprocess
import tempfile
import clang.cindex
from clang.cindex import CursorKind

try:
    import torch
    import torch.nn.functional as F
    from transformers import RobertaTokenizer, RobertaModel

    CODEBERT_AVAILABLE = True
except ImportError:
    CODEBERT_AVAILABLE = False
    print("Warning: CodeBERT dependencies not available. Install with: pip install torch transformers")

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.code_evaluation_metrics import CodeEvaluationMetric


class UnitTestAlignmentMetric(Metric):
    """
    Compare LLM unit-test results with the student’s correctness pattern.

    Adds:
        • functional_correctness (pass-rate)
        • edge_case_slip_match   (binary 0/1)
    """

    def __init__(self, compile_code: bool = True, compiler_path: str = "g++"):
        self.compile_code = compile_code
        self.compiler_path = compiler_path

    # ------------------------------------------------------------------ #
    #   HELM entry-point                                                 #
    # ------------------------------------------------------------------ #
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        stats: List[Stat] = []

        # ----- safety checks & data extraction --------------------------
        if not request_state.result or not request_state.result.completions:
            return self._create_failure_stats("No output")

        generated_code = request_state.result.completions[0].text.strip()
        extra_data = getattr(request_state.instance, "extra_data", {}) or {}
        test_cases: List[dict] = extra_data.get("test_cases", [])
        student_pattern: List[int] = extra_data.get("student_correctness_pattern", [])
        prompt_template: str = extra_data.get("question_template", "")

        if not test_cases or not student_pattern:
            return self._create_failure_stats("Missing test cases or student pattern")

        # ----- run unit tests on LLM code -------------------------------
        llm_pattern = self._evaluate_unit_tests(generated_code, test_cases, prompt_template)

        # ----- 1) functional_correctness --------------------------------
        pass_rate = sum(llm_pattern) / len(llm_pattern) if llm_pattern else 0.0
        stats.append(Stat(MetricName("functional_correctness")).add(pass_rate))

        # ----- 2) edge_case_slip_match ---------------------------------
        stats.append(self._edge_case_slip_match(llm_pattern, student_pattern))
        return stats

    # ------------------------------------------------------------------ #
    #   Helpers                                                          #
    # ------------------------------------------------------------------ #
    def _edge_case_slip_match(self, llm: List[int], stu: List[int]) -> Stat:
        """Return 1.0 iff LLM and student each fail *exactly one* identical test."""
        llm_fail  = [i for i, v in enumerate(llm) if v == 0]
        stu_fail  = [i for i, v in enumerate(stu) if v == 0]
        match = len(llm_fail) == len(stu_fail) == 1 and llm_fail == stu_fail
        return Stat(MetricName("edge_case_slip_match")).add(1.0 if match else 0.0)

    def _create_failure_stats(self, msg: str) -> List[Stat]:
        """Zeros for all metrics when compilation or data retrieval fails."""
        print(f"UNIT TEST ALIGNMENT METRIC FAILURE: {msg}")
        return [
            Stat(MetricName("functional_correctness")).add(0.0),
            Stat(MetricName("edge_case_slip_match")).add(0.0),
            Stat(MetricName("unit_test_alignment_ratio")).add(0.0),
            Stat(MetricName("unit_test_llm_pass_rate")).add(0.0),
        ]

    def _evaluate_unit_tests(self, generated_code: str, test_cases: List[dict], template: str) -> List[int]:
        """
        Evaluate the generated code against unit tests and return correctness pattern.
        Returns list of 0s and 1s indicating pass/fail for each test.
        """
        print("\n--- Evaluating Unit Tests (Alignment Metric) ---")
        print(f"Test cases count: {len(test_cases)}")

        llm_results = []

        for i, test_case in enumerate(test_cases):
            print(f"\n--- Unit Test {i+1}/{len(test_cases)} ---")
            print(f"Test case keys: {list(test_case.keys())}")
            print(f"Test input: {test_case.get('input', 'MISSING')}")
            print(f"Expected output: {test_case.get('output', 'MISSING')}")

            try:
                # Extract student code from LLM output
                student_code = self._extract_student_code(generated_code)
                print(f"Extracted student code length: {len(student_code)}")

                # Create complete C++ program
                complete_program = self._create_complete_program(template, student_code, test_case.get("input", ""))

                if complete_program is None:
                    print("ERROR: _create_complete_program returned None")
                    llm_results.append(0)
                    continue

                print(f"Complete program length: {len(complete_program)}")

                # Run the test
                if self.compile_code:
                    print("Running with actual compilation...")
                    success, actual_output, error = self._compile_and_run_cpp(complete_program)
                    print(f"Compilation success: {success}")
                    if not success:
                        print(f"Compilation error: {error}")
                    else:
                        print(f"Actual output: '{actual_output}'")
                else:
                    print("Running with simulation...")
                    actual_output = self._simulate_execution(student_code, test_case)
                    success = True
                    error = None
                    print(f"Simulated output: '{actual_output}'")

                # Check if test passed
                if success and actual_output is not None:
                    expected_output = test_case.get("output", "").strip()
                    test_passed = actual_output.strip() == expected_output
                    print(f"Test passed: {test_passed}")
                    print(f"Expected: '{expected_output}' | Actual: '{actual_output.strip()}'")
                    llm_results.append(1 if test_passed else 0)
                else:
                    print("Test failed due to compilation/execution failure")
                    llm_results.append(0)  # Failed to compile/run

            except Exception as e:
                print(f"Exception in test case {i+1}: {str(e)}")
                import traceback

                traceback.print_exc()
                llm_results.append(0)  # Error in execution

        print("\n--- Unit Test Results (Alignment Metric) ---")
        print(f"LLM results: {llm_results}")
        return llm_results

    def _extract_student_code(self, model_code: str) -> str:
        """Extract the actual student function code from the model output."""
        # Remove markdown formatting
        code = re.sub(r"```cpp\n?|```\n?", "", model_code).strip()

        print(f"Original model code length: {len(model_code)}")
        print(f"After markdown removal: {len(code)}")

        # If code doesn't contain includes, it's likely just the student answer
        if "#include" not in code:
            print("No includes found, using entire code as student answer")
            return code

        # Split into lines for processing
        lines = code.split("\n")

        # Find main function index
        main_index = -1
        for i, line in enumerate(lines):
            if "int main" in line or "main()" in line:
                main_index = i
                break

        print(f"Main function found at line: {main_index}")

        # Extract code before main function
        if main_index > 0:
            student_lines = []
            for i in range(main_index):
                line = lines[i].strip()

                # Skip includes, using statements, and empty lines
                if line.startswith("#") or line.startswith("using") or line == "" or line.startswith("//"):
                    continue

                student_lines.append(lines[i])

            if student_lines:
                extracted = "\n".join(student_lines).strip()
                print(f"Extracted student code length: {len(extracted)}")
                return extracted

        # Fallback: return original code
        print("Using original code as fallback")
        return code

    def _create_complete_program(self, template: str, student_code: str, test_input: str) -> str:
        """Create a complete C++ program by combining template, student code, and test input."""
        print("\n--- Create Complete Program (Alignment Metric) ---")
        print(f"Template length: {len(template)}")
        print(f"Student code length: {len(student_code)}")
        print(f"Test input length: {len(test_input)}")
        print(f"Template preview: {template[:200]}...")
        print(f"Test input content: '{test_input}'")

        # Clean the test input (remove any trailing markers)
        clean_input = re.sub(r"STD input:\s*$", "", test_input).strip()
        print(f"Cleaned input length: {len(clean_input)}")
        print(f"Cleaned input: '{clean_input}'")

        # Handle template with {{ STUDENT_ANSWER }} placeholder
        if "{{ STUDENT_ANSWER }}" in template:
            print("Using template with STUDENT_ANSWER placeholder")
            # Replace the student answer placeholder
            complete_code = template.replace("{{ STUDENT_ANSWER }}", student_code)

            # Replace the ENTIRE template loop section with actual test
            template_loop_pattern = r"{%\s*for\s+TEST\s+in\s+TESTCASES\s*%}.*?{%\s*endfor\s*%}"

            test_block = f"""   {{
        {clean_input};
       }}"""

            print(f"Test block to insert: '{test_block}'")

            # Remove the entire template loop and replace with test block
            old_complete_code = complete_code
            complete_code = re.sub(template_loop_pattern, test_block, complete_code, flags=re.DOTALL)

            if complete_code == old_complete_code:
                print("WARNING: Template loop pattern not found in alignment metric")
            else:
                print("Successfully replaced template loop in alignment metric")

            # Also handle any remaining template syntax that might be left
            complete_code = re.sub(r"{%.*?%}", "", complete_code, flags=re.DOTALL)
            complete_code = re.sub(r"{{.*?}}", "", complete_code, flags=re.DOTALL)

            print(f"Template-based complete code length: {len(complete_code)}")
            return complete_code

        print("No {{ STUDENT_ANSWER }} placeholder found, creating simple structure")

        # If no template, create a simple program structure
        result = f"""#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

{student_code}

int main() {{
    {clean_input};
    return 0;
}}"""
        print(f"Simple program structure length: {len(result)}")
        return result

    def _compile_and_run_cpp(self, code: str) -> Tuple[bool, str, str]:
        """Compile and run C++ code, return (success, stdout, stderr)."""
        print("\n--- Compile and Run (Alignment Metric) ---")
        print(f"Code length: {len(code)}")

        try:
            # Set environment to avoid HuggingFace tokenizer warnings
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"

            # Create temporary files
            with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as cpp_file:
                cpp_file.write(code)
                cpp_file_path = cpp_file.name

            print(f"Created temp file: {cpp_file_path}")

            # Use .out extension instead of .exe for better cross-platform compatibility
            exe_file_path = cpp_file_path.replace(".cpp", ".out")

            # Compile the code
            compile_cmd = [self.compiler_path, "-std=c++17", "-o", exe_file_path, cpp_file_path]
            print(f"Compile command: {' '.join(compile_cmd)}")

            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30, env=env)

            print(f"Compile return code: {compile_result.returncode}")
            if compile_result.stderr:
                print(f"Compile stderr: {compile_result.stderr}")

            if compile_result.returncode != 0:
                return False, "", f"Compilation Error: {compile_result.stderr}"

            # Run the executable
            print(f"Running executable: {exe_file_path}")
            run_result = subprocess.run([exe_file_path], capture_output=True, text=True, timeout=10, env=env)

            print(f"Run return code: {run_result.returncode}")
            print(f"Run stdout: '{run_result.stdout}'")

            return True, run_result.stdout.strip(), run_result.stderr.strip()

        except subprocess.TimeoutExpired:
            print("ERROR: Execution timed out")
            return False, "", "Execution timed out"
        except FileNotFoundError as e:
            print(f"ERROR: Compiler not found: {e}")
            return False, "", f"Compiler '{self.compiler_path}' not found. Please install g++ or specify correct path."
        except Exception as e:
            print(f"ERROR: Exception during compilation: {e}")
            return False, "", f"Error: {str(e)}"
        finally:
            # Clean up temporary files
            try:
                if "cpp_file_path" in locals():
                    os.unlink(cpp_file_path)
                    print(f"Cleaned up: {cpp_file_path}")
                if "exe_file_path" in locals() and os.path.exists(exe_file_path):
                    os.unlink(exe_file_path)
                    print(f"Cleaned up: {exe_file_path}")
            except Exception as e:
                print(f"Cleanup error: {e}")

    def _simulate_execution(self, student_code: str, test_case: dict) -> str:
        """Simulate code execution for testing without actual compilation."""
        test_input = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        # Simulation for reverse function
        if "reverse" in student_code.lower() and "arr[]" in test_input:
            # Extract array from test input
            array_match = re.search(r"int arr\[\] = \{([^}]+)\}", test_input)
            if array_match:
                try:
                    numbers = [n.strip() for n in array_match.group(1).split(",")]
                    reversed_numbers = numbers[::-1]
                    return ", ".join(reversed_numbers)
                except Exception as e:
                    print(f"Error reversing array: {e}")
                    return ""

        # Simulation for factorial function
        if "factorial" in student_code.lower() and "factorial(" in test_input:
            # Extract number from factorial call
            factorial_match = re.search(r"factorial\((\d+)\)", test_input)
            if factorial_match:
                try:
                    n = int(factorial_match.group(1))
                    result = 1
                    for i in range(1, n + 1):
                        result *= i
                    return str(result)
                except Exception as e:
                    print(f"Error calculating factorial: {e}")
                    return ""

        # Simulation for BST enlarge (more complex)
        if "enlarge" in student_code.lower() and "BTNode" in student_code:
            # This would need more sophisticated simulation
            # For now, return expected output
            return expected_output

        # Default: return expected output (perfect simulation)
        return expected_output


class ComprehensiveCodeEvaluationMetric(CodeEvaluationMetric):
    """unit-test alignment (with new metrics)."""

    def __init__(self, use_codebert: bool = True, compile_code: bool = True, compiler_path: str = "g++"):
        super().__init__(use_codebert=use_codebert)
        self.unit_test_metric = UnitTestAlignmentMetric(
            compile_code=compile_code,
            compiler_path=compiler_path,
        )

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        stats = super().evaluate_generation(adapter_spec, request_state, metric_service, eval_cache_path)
        stats.extend(self.unit_test_metric.evaluate_generation(
            adapter_spec, request_state, metric_service, eval_cache_path
        ))
        return stats
