from typing import Any, Dict, List, Optional
import ast
import pandas as pd
import re

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
    """Utility class for calculating AST edit distances between code snippets."""
    
    def calculate_ast_distance(self, code1: str, code2: str) -> float:
        """
        Calculate normalized AST edit distance between two code snippets.
        Returns a value between 0 and 1, where 0 means identical and 1 means completely different.
        """
        try:
            # Parse both code snippets into ASTs
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            # Convert ASTs to comparable structures
            nodes1 = self._extract_ast_features(tree1)
            nodes2 = self._extract_ast_features(tree2)
            
            # Calculate edit distance using simple node comparison
            distance = self._calculate_edit_distance(nodes1, nodes2)
            
            # Normalize by the maximum possible distance
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
        """Compute fixed-size embedding vector for source code using CodeBERT."""
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
        ast_similarity = 1.0 - ast_distance
        perfect_match = 1.0 if ast_distance == 0.0 else 0.0
        close_match = 1.0 if ast_distance <= 0.1 else 0.0
        
        return [
            Stat(MetricName("ast_distance")).add(ast_distance),
            Stat(MetricName("ast_similarity")).add(ast_similarity),
            Stat(MetricName("ast_perfect_match")).add(perfect_match),
            Stat(MetricName("ast_close_match")).add(close_match)
        ]
    
    def _create_codebert_stats(self, codebert_similarity: float) -> List[Stat]:
        """Create CodeBERT-based statistics."""
        # Convert similarity to distance for consistency
        codebert_distance = 1.0 - codebert_similarity
        high_similarity = 1.0 if codebert_similarity >= 0.9 else 0.0
        medium_similarity = 1.0 if codebert_similarity >= 0.7 else 0.0
        
        return [
            Stat(MetricName("codebert_similarity")).add(codebert_similarity),
            Stat(MetricName("codebert_distance")).add(codebert_distance),
            Stat(MetricName("codebert_high_similarity")).add(high_similarity),
            Stat(MetricName("codebert_medium_similarity")).add(medium_similarity)
        ]


class AdvancedCodeEvaluationMetric(CodeEvaluationMetric):
    """Extended code evaluation metric with additional analyses."""
    
    def __init__(self, use_codebert: bool = True):
        super().__init__(use_codebert=use_codebert)
    
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate with additional code quality metrics."""
        # Get base AST and CodeBERT metrics
        stats = super().evaluate_generation(adapter_spec, request_state, metric_service, eval_cache_path)
        
        if not request_state.result or not request_state.result.completions:
            return stats
        
        generated_code = request_state.result.completions[0].text.strip()
        
        # Add code quality metrics
        try:
            compilation_success = self._check_compilation(generated_code)
            code_complexity = self._calculate_complexity(generated_code)
            
            stats.append(Stat(MetricName("compilation_success")).add(1.0 if compilation_success else 0.0))
            stats.append(Stat(MetricName("code_complexity")).add(code_complexity))
            
        except Exception:
            stats.append(Stat(MetricName("compilation_success")).add(0.0))
            stats.append(Stat(MetricName("code_complexity")).add(0.0))
        
        return stats
    
    def _check_compilation(self, code: str) -> bool:
        """Check if code can be parsed successfully."""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError):
            return False
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate a simple complexity metric based on AST node count."""
        try:
            tree = ast.parse(code)
            node_count = len(list(ast.walk(tree)))
            # Normalize by a reasonable baseline (e.g., 10 nodes)
            return min(node_count / 10.0, 5.0)  # Cap at 5x baseline complexity
        except (SyntaxError, ValueError):
            return 0.0


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