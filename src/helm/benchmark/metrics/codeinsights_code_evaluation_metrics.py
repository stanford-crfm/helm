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


class ASTAnalyzer:
    """Class for calculating AST edit distances between two C++ code snippets using libclang."""

    def __init__(self, clang_lib_path: str = ""):
        """
        If libclang isn't on your LD_LIBRARY_PATH, pass its full path here.
        e.g. '/usr/lib/llvm-14/lib/libclang.so'
        """
        if clang_lib_path:
            clang.cindex.Config.set_library_file(clang_lib_path)
        self.index = clang.cindex.Index.create()

    def calculate_ast_distance(self, code1: str, code2: str) -> float:
        """
        Calculate normalized AST edit distance between two C++ code snippets.
        Returns a float in [0,1], where 0 means identical ASTs and 1 means completely different
        (or a parse failure).
        """
        try:
            tu1 = self.index.parse(path="code1.cpp", args=["-std=c++17"], unsaved_files=[("code1.cpp", code1)])
            tu2 = self.index.parse(path="code2.cpp", args=["-std=c++17"], unsaved_files=[("code2.cpp", code2)])

            nodes1: List[str] = self._extract_ast_features(tu1.cursor)
            nodes2: List[str] = self._extract_ast_features(tu2.cursor)

            distance = self._calculate_edit_distance(nodes1, nodes2)
            max_nodes = max(len(nodes1), len(nodes2))
            if max_nodes == 0:
                return 0.0
            return min(distance / max_nodes, 1.0)

        except Exception:
            # any parse error or clang error → max distance
            return 1.0

    def _extract_ast_features(self, node: clang.cindex.Cursor) -> List[str]:
        """Recursively walk Clang AST, appending feature strings to feats."""
        feats: List[str] = []
        # record the node kind
        feats.append(node.kind.name)

        # some node-specific details
        if node.kind == CursorKind.FUNCTION_DECL:
            feats.append(f"Function:{node.spelling}")
        elif node.kind == CursorKind.DECL_REF_EXPR:
            feats.append(f"Ref:{node.spelling}")
        elif node.kind in (
            CursorKind.INTEGER_LITERAL,
            CursorKind.FLOATING_LITERAL,
            CursorKind.STRING_LITERAL,
            CursorKind.CHARACTER_LITERAL,
        ):
            # get literal token text
            tokens = list(node.get_tokens())
            if tokens:
                feats.append(f"Literal:{tokens[0].spelling}")

        # recurse
        for child in node.get_children():
            feats = feats + self._extract_ast_features(child)

        return feats

    def _calculate_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Classic Levenshtein edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  # delete  # insert  # substitute
        return dp[m][n]


class CodeBERTAnalyzer:
    """Utility class for calculating semantic code similarity using CodeBERT."""

    def __init__(self):
        if not CODEBERT_AVAILABLE:
            raise ImportError("CodeBERT dependencies not available. Install with: pip install torch transformers")

        # Initialize CodeBERT model and tokenizer
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name)
        self.model.eval()  # Set to evaluation mode

    def strip_code_fence(self, code: str) -> str:
        """Remove code fence markers from code strings."""
        # Remove ```python, ```cpp, ``` etc.
        code = re.sub(r"^```\w*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)
        return code.strip()

    def get_code_embedding(self, code: str, max_length: int = 512) -> torch.Tensor:
        """Compute fixed-size embedding vector for code using CodeBERT."""
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")

        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state

        # Apply attention mask and average valid token embeddings
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embedding = summed / counts

        return embedding.squeeze(0)

    def calculate_embedding_similarity(self, code1: str, code2: str) -> float:
        """Calculate cosine similarity between code embeddings."""
        clean_code1 = self.strip_code_fence(code1)
        clean_code2 = self.strip_code_fence(code2)

        emb1 = self.get_code_embedding(clean_code1)
        emb2 = self.get_code_embedding(clean_code2)

        cosine_sim = F.cosine_similarity(emb1, emb2, dim=0).item()
        return cosine_sim


class CodeEvaluationMetric(Metric):
    """Metric for evaluating code generation quality using AST analysis and CodeBERT similarity."""

    def __init__(self, use_codebert: bool = True):
        self.ast_analyzer = ASTAnalyzer()
        self.use_codebert = use_codebert and CODEBERT_AVAILABLE

        if self.use_codebert:
            try:
                self.codebert_analyzer = CodeBERTAnalyzer()
            except Exception as e:
                print(f"Warning: Failed to initialize CodeBERT analyzer: {e}")
                self.use_codebert = False

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate a single generated code snippet."""
        stats = []

        # Get the generated code from the request state
        if not request_state.result or not request_state.result.completions:
            return self._create_default_stats(1.0)  # Maximum distance for no output

        generated_code = request_state.result.completions[0].text.strip()

        # Get the ground truth from the instance references
        if not request_state.instance.references:
            return self._create_default_stats(1.0)

        ground_truth = request_state.instance.references[0].output.text.strip()

        # Calculate AST distance
        if "Error:" in generated_code or not generated_code:
            ast_distance = 1.0  # Maximum distance for errors or empty output
        else:
            try:
                ast_distance = self.ast_analyzer.calculate_ast_distance(generated_code, ground_truth)
            except Exception:
                ast_distance = 1.0  # Maximum distance for parsing errors

        # Create AST-based statistics
        stats.extend(self._create_ast_stats(ast_distance))

        # Calculate CodeBERT similarity if available
        if self.use_codebert:
            try:
                codebert_similarity = self.codebert_analyzer.calculate_embedding_similarity(
                    generated_code, ground_truth
                )
                stats.extend(self._create_codebert_stats(codebert_similarity))
            except Exception as e:
                print(f"Warning: CodeBERT similarity calculation failed: {e}")
                # Add default CodeBERT stats for failed calculations
                stats.extend(self._create_codebert_stats(0.0))

        return stats

    def _create_default_stats(self, distance: float) -> List[Stat]:
        """Create default statistics for error cases."""
        stats = self._create_ast_stats(distance)
        if self.use_codebert:
            stats.extend(self._create_codebert_stats(0.0))
        return stats

    def _create_ast_stats(self, ast_distance: float) -> List[Stat]:
        """Create AST-based statistics."""
        return [Stat(MetricName("ast_distance")).add(ast_distance)]

    def _create_codebert_stats(self, codebert_similarity: float) -> List[Stat]:
        """Create CodeBERT-based statistics."""
        return [Stat(MetricName("codebert_similarity")).add(codebert_similarity)]


class AdvancedCodeEvaluationMetric(CodeEvaluationMetric):
    """Extended code evaluation metric with additional analyses"""

    def __init__(self, use_codebert: bool = True):
        super().__init__(use_codebert=use_codebert)


class UnitTestAlignmentMetric(Metric):
    """Metric for evaluating C++ code generation by comparing unit test results with student correctness pattern."""

    def __init__(self, compile_code: bool = True, compiler_path: str = "g++"):
        """
        Initialize the unit test alignment metric.

        Args:
            compile_code: Whether to actually compile and run code (True) or simulate (False)
            compiler_path: Path to the C++ compiler (default: "g++")
        """
        self.compile_code = compile_code
        self.compiler_path = compiler_path

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate LLM-generated code by comparing unit test results with student correctness pattern."""
        print("\n=== UNIT TEST ALIGNMENT METRIC DEBUG ===")
        print(f"Instance ID: {getattr(request_state.instance, 'id', 'UNKNOWN')}")

        stats = []

        # Get the generated code from the request state
        if not request_state.result or not request_state.result.completions:
            print("ERROR: No output generated")
            return self._create_failure_stats("No output generated")

        generated_code = request_state.result.completions[0].text.strip()
        print(f"Generated code length: {len(generated_code)}")
        print(f"Generated code preview: {generated_code[:200]}...")

        # Get test cases and student correctness pattern from instance extra_data
        if not hasattr(request_state.instance, "extra_data") or not request_state.instance.extra_data:
            print("ERROR: No extra_data available")
            return self._create_failure_stats("No test data available")

        extra_data = request_state.instance.extra_data
        print(f"Extra data keys: {list(extra_data.keys())}")

        test_cases = extra_data.get("test_cases", [])
        student_correctness_pattern = extra_data.get("student_correctness_pattern", [])
        prompt_template = extra_data.get("question_template", "")
        question_name = extra_data.get("question_name", "UNKNOWN")

        print(f"Question name: {question_name}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"Student correctness pattern: {student_correctness_pattern}")
        print(f"Template length: {len(prompt_template)}")

        if not test_cases or not student_correctness_pattern:
            print("ERROR: Missing test cases or student correctness pattern")
            return self._create_failure_stats("Missing test cases or student correctness pattern")

        print(f"First test case preview: {test_cases[0] if test_cases else 'NONE'}")

        # Run unit tests and get LLM correctness pattern
        llm_correctness_pattern = self._evaluate_unit_tests(generated_code, test_cases, prompt_template)

        print(f"LLM correctness pattern: {llm_correctness_pattern}")

        # Compare patterns and calculate alignment metrics
        alignment_stats = self._calculate_alignment_metrics(llm_correctness_pattern, student_correctness_pattern)

        stats.extend(alignment_stats)
        print(f"Final alignment stats: {[stat.name.name for stat in stats]}")
        print("=== END UNIT TEST ALIGNMENT DEBUG ===\n")

        return stats

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

    def _calculate_alignment_metrics(self, llm_pattern: List[int], student_pattern: List[int]) -> List[Stat]:
        """
        Calculate alignment metrics between LLM and student correctness patterns.
        """
        # Ensure patterns have same length (pad with 0s if needed)
        max_length = max(len(llm_pattern), len(student_pattern))
        llm_padded = llm_pattern + [0] * (max_length - len(llm_pattern))
        student_padded = student_pattern + [0] * (max_length - len(student_pattern))

        # Calculate alignment metrics
        total_tests = max_length
        exact_matches = sum(1 for i in range(total_tests) if llm_padded[i] == student_padded[i])

        # Alignment ratio (percentage of matching tests)
        alignment_ratio = exact_matches / total_tests if total_tests > 0 else 0.0

        # Calculate LLM and student pass rates
        llm_pass_rate = sum(llm_padded) / total_tests if total_tests > 0 else 0.0
        student_pass_rate = sum(student_padded) / total_tests if total_tests > 0 else 0.0

        print(f"Alignment calculation: {exact_matches}/{total_tests} = {alignment_ratio}")

        return [
            Stat(MetricName("unit_test_alignment_ratio")).add(alignment_ratio),
            Stat(MetricName("unit_test_llm_pass_rate")).add(llm_pass_rate),
            Stat(MetricName("unit_test_student_pass_rate")).add(student_pass_rate),
        ]

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

    def _create_failure_stats(self, error_message: str) -> List[Stat]:
        """Create default statistics for failure cases."""
        print(f"UNIT TEST ALIGNMENT METRIC FAILURE: {error_message}")
        return [
            Stat(MetricName("unit_test_alignment_ratio")).add(0.0),
            Stat(MetricName("unit_test_llm_pass_rate")).add(0.0),
        ]


class ComprehensiveCodeEvaluationMetric(CodeEvaluationMetric):
    """Comprehensive metric combining AST, CodeBERT, and unit test alignment."""

    def __init__(self, use_codebert: bool = True, compile_code: bool = True, compiler_path: str = "g++"):
        super().__init__(use_codebert=use_codebert)
        self.unit_test_metric = UnitTestAlignmentMetric(compile_code=compile_code, compiler_path=compiler_path)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate with AST, CodeBERT, and unit test alignment metrics."""

        # Get base AST and CodeBERT metrics
        stats = super().evaluate_generation(adapter_spec, request_state, metric_service, eval_cache_path)

        # Add unit test alignment metrics
        unit_test_stats = self.unit_test_metric.evaluate_generation(
            adapter_spec, request_state, metric_service, eval_cache_path
        )
        stats.extend(unit_test_stats)
        return stats


# Legacy method for batch evaluation (if needed for backward compatibility)
def evaluate_ast_distances_batch(results: Dict, analyzer: ASTAnalyzer) -> pd.DataFrame:
    """
    Legacy batch evaluation method for AST distances.
    This can be used outside of HELM if needed.
    """
    all_rows = []

    for student_id, info in results.items():
        ground_truth = info["ground_truth"]

        for model_name, generated_code in info["outputs"].items():
            if "Error:" in generated_code:
                normalized_distance = 1.0  # Maximum distance for errors
            else:
                try:
                    normalized_distance = analyzer.calculate_ast_distance(generated_code, ground_truth)
                except Exception:
                    normalized_distance = 1.0

            all_rows.append(
                {
                    "student_id": student_id,
                    "question_id": info["question_id"],
                    "model": model_name,
                    "normalized_distance": normalized_distance,
                }
            )

    df_long = pd.DataFrame(all_rows)
    df_ast = df_long.pivot(
        index=["student_id", "question_id"], columns="model", values="normalized_distance"
    ).reset_index()

    return df_ast
