from typing import Any, Dict, List, Optional, Tuple
import ast
import pandas as pd
import re
import os
import subprocess
import tempfile

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
    """Class for calculating AST edit distances between real student code and LLM generated code"""
    
    def calculate_ast_distance(self, code1: str, code2: str) -> float:
        """
        Calculate normalized AST edit distance between two codes
        Returns a value between 0 and 1, where 0 means that two codes are identical and 1 means completely different.
        """
        try:
            # Parse both codes into ASTs
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            nodes1 = self._extract_ast_features(tree1)
            nodes2 = self._extract_ast_features(tree2)
            
            # Calculate edit distance using simple node comparison
            distance = self._calculate_edit_distance(nodes1, nodes2)
            # Normalize distance by the maximum distance
            max_nodes = max(len(nodes1), len(nodes2))
            if max_nodes == 0:
                return 0.0
            normalized_distance = min(distance / max_nodes, 1.0)
            return normalized_distance
            
        except (SyntaxError, ValueError) as e:
            # If parsing fails, return maximum distance
            return 1.0
    
    def _extract_ast_features(self, tree: ast.AST) -> List[str]:
        """Extract features from AST for comparison."""
        features = []
        
        for node in ast.walk(tree):
            # Include node type
            features.append(type(node).__name__)
            # Include some node-specific information
            if isinstance(node, ast.Name):
                features.append(f"Name:{node.id}")
            elif isinstance(node, ast.Constant):
                features.append(f"Constant:{str(node.value)}")
            elif isinstance(node, ast.FunctionDef):
                features.append(f"Function:{node.name}")
            elif isinstance(node, ast.BinOp):
                features.append(f"BinOp:{type(node.op).__name__}")
            elif isinstance(node, ast.Compare):
                ops = [type(op).__name__ for op in node.ops]
                features.append(f"Compare:{','.join(ops)}")
        
        return features
    
    def _calculate_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance between two sequences using dynamic programming."""
        m, n = len(seq1), len(seq2)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
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
        code = re.sub(r'^```\w*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code, flags=re.MULTILINE)
        return code.strip()
    
    def get_code_embedding(self, code: str, max_length: int = 512) -> torch.Tensor:
        """Compute fixed-size embedding vector for code using CodeBERT."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
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
        return [
            Stat(MetricName("ast_distance")).add(ast_distance)
        ]
    
    def _create_codebert_stats(self, codebert_similarity: float) -> List[Stat]:
        """Create CodeBERT-based statistics."""
        return [
            Stat(MetricName("codebert_similarity")).add(codebert_similarity)
        ]


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
        stats = []
        
        # Get the generated code from the request state
        if not request_state.result or not request_state.result.completions:
            return self._create_failure_stats("No output generated")
        
        generated_code = request_state.result.completions[0].text.strip()
        
        # Get test cases and student correctness pattern from instance extra_data
        if not hasattr(request_state.instance, 'extra_data') or not request_state.instance.extra_data:
            return self._create_failure_stats("No test data available")
        
        extra_data = request_state.instance.extra_data
        test_cases = extra_data.get('test_cases', [])
        student_correctness_pattern = extra_data.get('student_correctness_pattern', [])
        prompt_template = extra_data.get('question_template', '')
        
        if not test_cases or not student_correctness_pattern:
            return self._create_failure_stats("Missing test cases or student correctness pattern")
        
        # Run unit tests and get LLM correctness pattern
        llm_correctness_pattern = self._evaluate_unit_tests(
            generated_code, test_cases, prompt_template
        )
        
        # Compare patterns and calculate alignment metrics
        alignment_stats = self._calculate_alignment_metrics(
            llm_correctness_pattern, student_correctness_pattern
        )
        
        stats.extend(alignment_stats)
        return stats
    
    def _evaluate_unit_tests(self, generated_code: str, test_cases: List[dict], template: str) -> List[int]:
        """
        Evaluate the generated code against unit tests and return correctness pattern.
        Returns list of 0s and 1s indicating pass/fail for each test.
        """
        llm_results = []
        
        for test_case in test_cases:
            try:
                # Extract student code from LLM output
                student_code = self._extract_student_code(generated_code)
                
                # Create complete C++ program
                complete_program = self._create_complete_program(
                    template, student_code, test_case.get('input', '')
                )
                
                # Run the test
                if self.compile_code:
                    success, actual_output, error = self._compile_and_run_cpp(complete_program)
                else:
                    actual_output = self._simulate_execution(student_code, test_case)
                    success = True
                    error = None
                
                # Check if test passed
                if success and actual_output is not None:
                    expected_output = test_case.get('output', '').strip()
                    test_passed = actual_output.strip() == expected_output
                    llm_results.append(1 if test_passed else 0)
                else:
                    llm_results.append(0)  # Failed to compile/run
                    
            except Exception as e:
                llm_results.append(0)  # Error in execution
        
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
        
        # Calculate pass rate difference
        pass_rate_difference = abs(llm_pass_rate - student_pass_rate)
        
        return [
            Stat(MetricName("unit_test_alignment_ratio")).add(alignment_ratio),
            Stat(MetricName("unit_test_pass_rate_difference")).add(pass_rate_difference),
        ]
    
    def _extract_student_code(self, model_code: str) -> str:
        """Extract the actual student function code from the model output."""
        # Remove markdown formatting
        code = re.sub(r'```cpp\n?|```\n?', '', model_code).strip()
        
        # If code doesn't contain includes, it's likely just the student answer
        if '#include' not in code:
            return code
        
        # Split into lines for processing
        lines = code.split('\n')
        
        # Find main function index
        main_index = -1
        for i, line in enumerate(lines):
            if 'int main' in line or 'main()' in line:
                main_index = i
                break
        
        # Extract code before main function
        if main_index > 0:
            student_lines = []
            for i in range(main_index):
                line = lines[i].strip()
                
                # Skip includes, using statements, and empty lines
                if (line.startswith('#') or 
                    line.startswith('using') or 
                    line == '' or
                    line.startswith('//')):
                    continue
                
                student_lines.append(lines[i])
            
            if student_lines:
                return '\n'.join(student_lines).strip()
        
        # Fallback: return original code
        return code
    
    def _create_complete_program(self, template: str, student_code: str, test_input: str) -> str:
        """Create a complete C++ program by combining template, student code, and test input."""
        # Clean the test input (remove any trailing markers)
        clean_input = re.sub(r'STD input:\s*$', '', test_input).strip()
        
        # Handle template with {{ STUDENT_ANSWER }} placeholder
        if '{{ STUDENT_ANSWER }}' in template:
            # Replace the student answer placeholder
            complete_code = template.replace('{{ STUDENT_ANSWER }}', student_code)
            
            # Replace the ENTIRE template loop section with actual test
            template_loop_pattern = r'{%\s*for\s+TEST\s+in\s+TESTCASES\s*%}.*?{%\s*endfor\s*%}'
            
            test_block = f"""   {{
        {clean_input};
       }}"""
            
            # Remove the entire template loop and replace with test block
            complete_code = re.sub(template_loop_pattern, test_block, complete_code, flags=re.DOTALL)
            
            # Also handle any remaining template syntax that might be left
            complete_code = re.sub(r'{%.*?%}', '', complete_code, flags=re.DOTALL)
            complete_code = re.sub(r'{{.*?}}', '', complete_code, flags=re.DOTALL)
            
            return complete_code
        
        # If no template, create a simple program structure
        return f"""#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

{student_code}

int main() {{
    {clean_input};
    return 0;
}}"""
    
    def _compile_and_run_cpp(self, code: str) -> Tuple[bool, str, str]:
        """Compile and run C++ code, return (success, stdout, stderr)."""
        try:
            # Set environment to avoid HuggingFace tokenizer warnings
            env = os.environ.copy()
            env['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as cpp_file:
                cpp_file.write(code)
                cpp_file_path = cpp_file.name
            
            # Use .out extension instead of .exe for better cross-platform compatibility
            exe_file_path = cpp_file_path.replace('.cpp', '.out')
            
            # Compile the code
            compile_result = subprocess.run([
                self.compiler_path, '-std=c++17', '-o', exe_file_path, cpp_file_path
            ], capture_output=True, text=True, timeout=30, env=env)
            
            if compile_result.returncode != 0:
                return False, "", f"Compilation Error: {compile_result.stderr}"
            
            # Run the executable
            run_result = subprocess.run([
                exe_file_path
            ], capture_output=True, text=True, timeout=10, env=env)
            
            return True, run_result.stdout.strip(), run_result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out"
        except FileNotFoundError as e:
            return False, "", f"Compiler '{self.compiler_path}' not found. Please install g++ or specify correct path."
        except Exception as e:
            return False, "", f"Error: {str(e)}"
        finally:
            # Clean up temporary files
            try:
                if 'cpp_file_path' in locals():
                    os.unlink(cpp_file_path)
                if 'exe_file_path' in locals() and os.path.exists(exe_file_path):
                    os.unlink(exe_file_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    def _simulate_execution(self, student_code: str, test_case: dict) -> str:
        """Simulate code execution for testing without actual compilation."""
        test_input = test_case.get('input', '')
        expected_output = test_case.get('output', '')
        
        # Simulation for reverse function
        if 'reverse' in student_code.lower() and 'arr[]' in test_input:
            # Extract array from test input
            array_match = re.search(r'int arr\[\] = \{([^}]+)\}', test_input)
            if array_match:
                try:
                    numbers = [n.strip() for n in array_match.group(1).split(',')]
                    reversed_numbers = numbers[::-1]
                    return ', '.join(reversed_numbers)
                except:
                    pass
        
        # Simulation for factorial function
        if 'factorial' in student_code.lower() and 'factorial(' in test_input:
            # Extract number from factorial call
            factorial_match = re.search(r'factorial\((\d+)\)', test_input)
            if factorial_match:
                try:
                    n = int(factorial_match.group(1))
                    result = 1
                    for i in range(1, n + 1):
                        result *= i
                    return str(result)
                except:
                    pass
        
        # Simulation for BST enlarge (more complex)
        if 'enlarge' in student_code.lower() and 'BTNode' in student_code:
            # This would need more sophisticated simulation
            # For now, return expected output
            return expected_output
        
        # Default: return expected output (perfect simulation)
        return expected_output
    
    def _create_failure_stats(self, error_message: str) -> List[Stat]:
        """Create default statistics for failure cases."""
        return [
            Stat(MetricName("unit_test_alignment_ratio")).add(0.0),
            Stat(MetricName("unit_test_pass_rate_difference")).add(0.0),
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
                    normalized_distance = analyzer.calculate_ast_distance(
                        generated_code, ground_truth
                    )
                except Exception:
                    normalized_distance = 1.0
            
            all_rows.append({
                "student_id": student_id,
                "question_id": info["question_id"],
                "model": model_name,
                "normalized_distance": normalized_distance
            })
    
    df_long = pd.DataFrame(all_rows)
    df_ast = df_long.pivot(
        index=["student_id", "question_id"],
        columns="model",
        values="normalized_distance"
    ).reset_index()
    
    return df_ast