from typing import Dict, List
import pandas as pd
import re
import os
import subprocess
import tempfile
import clang.cindex
from clang.cindex import CursorKind
from Levenshtein import ratio as levenshtein_distance_ratio

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
from helm.benchmark.metrics.codeinsights_correct_code_metrics import (
    CodeInsightsFunctionalCorrectnessMetric,
    CPPEvaluator,
)


def _cpp_to_asm(src: str, compiler: str = "g++") -> str:
    """Return the assembly text for `src`, or '' if the compile fails."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(src)
        cpp_path = f.name
    asm_path = cpp_path.replace(".cpp", ".s")
    try:
        subprocess.run(
            [compiler, "-std=c++17", "-S", "-o", asm_path, cpp_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        temp_file_name = os.path.basename(asm_path)
        with open(asm_path, "r") as fh:
            asm_code = fh.read()
            asm_code = asm_code.replace(temp_file_name, "asm_output.cpp")  # Normalize file name in output
            return asm_code

    except subprocess.CalledProcessError as e:
        print("⚠️  Assembly compilation failed:", e.stderr[:200])
        return ""
    finally:
        try:
            os.unlink(cpp_path)
            if os.path.exists(asm_path):
                os.unlink(asm_path)
        except Exception:
            pass


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

            return levenshtein_distance_ratio(nodes1, nodes2)

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
        else:
            # for other nodes, just use the spelling if available
            if node.spelling:
                feats.append(f"Other:{node.spelling}")

        # recurse
        for child in node.get_children():
            feats = feats + self._extract_ast_features(child)

        return feats


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


class CodeInsightsCodeEvaluationMetric(Metric):
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
            return self._create_default_stats(0.5)

        generated_code = request_state.result.completions[0].text.strip()
        generated_code = self._extract_student_code(generated_code)

        # Get the ground truth from the instance references
        if not request_state.instance.references:
            return self._create_default_stats(0.5)

        ground_truth = request_state.instance.references[0].output.text.strip()

        # Calculate AST distance
        if not generated_code or "Error:" in generated_code:
            ast_distance = 1.0
        else:
            try:
                ast_distance = self.ast_analyzer.calculate_ast_distance(generated_code, ground_truth)
            except Exception:
                ast_distance = 1.0

        # Create AST-based statistics
        stats.extend(self._create_ast_stats(ast_distance))

        # Calculate assembly distance
        gen_asm = _cpp_to_asm(generated_code)
        truth_asm = _cpp_to_asm(ground_truth)
        if gen_asm == "" or truth_asm == "":
            asm_distance = 1.0
        else:
            asm_distance = levenshtein_distance_ratio(gen_asm, truth_asm)

        # Create assembly-based statistics
        stats.extend(self._create_asm_stats(asm_distance))

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

        print(f"[Final extracted code length] {len(code)}")
        print(f"[Code preview]\n{code[:300]}...\n")
        return code

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

    def _create_asm_stats(self, asm_distance: float) -> List[Stat]:
        """Create assembly-based statistics."""
        return [Stat(MetricName("asm_distance")).add(asm_distance)]


class AdvancedCodeEvaluationMetric(CodeInsightsCodeEvaluationMetric):
    """Extended code evaluation metric with additional analyses"""

    def __init__(self, use_codebert: bool = True):
        super().__init__(use_codebert=use_codebert)


class UnitTestAlignmentMetric(CodeInsightsFunctionalCorrectnessMetric):
    """Metric for evaluating C++ code generation by comparing unit test results with student correctness pattern."""

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
            Stat(MetricName("unittest_alignment_ratio")).add(alignment_ratio),
            Stat(MetricName("unittest_llm_pass_rate")).add(llm_pass_rate),
            Stat(MetricName("unittest_student_pass_rate")).add(student_pass_rate),
        ]

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
        student_correctness_pattern = extra_data.get("student_correctness_pattern", [])
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
        llm_correctness_pattern = evaluator.evaluate(generated_code)["testcases"]
        print(f"LLM correctness pattern: {llm_correctness_pattern}")

        # Compare patterns and calculate alignment metrics
        alignment_stats = self._calculate_alignment_metrics(llm_correctness_pattern, student_correctness_pattern)

        print(f"Final alignment stats: {[stat.name.name for stat in alignment_stats]}")
        print("=== END UNIT TEST ALIGNMENT DEBUG ===\n")

        return alignment_stats

    def _create_failure_stats(self, error_message: str) -> List[Stat]:
        """Create default statistics for failure cases."""
        print(f"UNIT TEST ALIGNMENT METRIC FAILURE: {error_message}")
        return [
            Stat(MetricName("unittest_alignment_ratio")).add(0.0),
            Stat(MetricName("unittest_llm_pass_rate")).add(0.0),
            Stat(MetricName("unittest_student_pass_rate")).add(0.0),
        ]


class CodeInsightsComprehensiveCodeEvaluationMetric(CodeInsightsCodeEvaluationMetric):
    """Comprehensive metric combining AST, CodeBERT, and unit test alignment."""

    def __init__(self, use_codebert: bool = True):
        super().__init__(use_codebert=use_codebert)
        self.unittest_metric = UnitTestAlignmentMetric()

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
        unittest_stats = self.unittest_metric.evaluate_generation(
            adapter_spec, request_state, metric_service, eval_cache_path
        )
        stats.extend(unittest_stats)
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
