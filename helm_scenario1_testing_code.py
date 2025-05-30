import pandas as pd
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Third-party imports
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import google.generativeai as genai
from google import genai
from google.genai import types
from clang import cindex
from zss import Node, simple_distance


class Config:
    """Configuration class for API keys and model settings."""
    
    # API Keys - In production, use environment variables or secure config
    OPENAI_API_KEY = "API_KEY"
    ANTHROPIC_API_KEY = "API_KEY"
    DEEPSEEK_API_KEY = "API_KEY"
    GOOGLE_API_KEY = "API_KEY"
    
    # Model configurations
    OPENAI_MODEL = "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
    DEEPSEEK_MODEL = "deepseek-chat"
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Code embedding model
    CODEBERT_MODEL = "microsoft/codebert-base"
    
    # Data URL
    DATA_URL = "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/sample_fifty_student_full.csv"


class LLMClients:
    """Manages API clients for different LLM providers."""
    
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.deepseek_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com"
        )
        self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.gemini_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        
        # System prompt for all models
        self.system_prompt = (
            "You are an undergraduate student in a foundational C++ programming course in Vietnam. "
            "Your task is to generate code for a given question."
        )


class CodeGenerator:
    """Handles code generation from different LLM providers."""
    
    def __init__(self, clients: LLMClients):
        self.clients = clients
    
    def call_openai(self, prompt: str) -> str:
        """Generate code using OpenAI GPT."""
        try:
            response = self.clients.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.clients.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    def call_anthropic(self, prompt: str) -> str:
        """Generate code using Anthropic Claude."""
        try:
            full_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
            response = self.clients.anthropic_client.messages.create(
                model=Config.ANTHROPIC_MODEL,
                temperature=0,
                max_tokens=2000,
                system=self.clients.system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {e}"
    
    def call_deepseek(self, prompt: str) -> str:
        """Generate code using DeepSeek."""
        try:
            response = self.clients.deepseek_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self.clients.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    def call_gemini(self, prompt: str) -> str:
        """Generate code using Google Gemini."""
        try:
            response = self.clients.gemini_client.models.generate_content(
                model=Config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self.clients.system_prompt
                ),
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def generate_all(self, prompt: str) -> Dict[str, str]:
        """Generate code using all available models."""
        runners = {
            "OpenAI": self.call_openai,
            "Anthropic": self.call_anthropic,
            "DeepSeek": self.call_deepseek,
            "Gemini": self.call_gemini,
        }
        
        results = {}
        for model_name, runner_fn in runners.items():
            results[model_name] = runner_fn(prompt)
        
        return results


class CodeAnalyzer:
    """Handles code analysis including AST parsing and embeddings."""
    
    def __init__(self):
        # Initialize CodeBERT for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(Config.CODEBERT_MODEL)
        self.model = AutoModel.from_pretrained(Config.CODEBERT_MODEL)
        self.model.eval()
    
    @staticmethod
    def strip_code_fence(code: str) -> str:
        """Remove markdown code fence markers from code."""
        code = re.sub(r"^```(?:\w+)?\n", "", code)
        code = re.sub(r"\n```$", "", code)
        return code
    
    @staticmethod
    def parse_ast(code: str, filename: str = 'tmp.cpp'):
        """Parse C++ code into a clang AST."""
        index = cindex.Index.create()
        tu = index.parse(
            filename,
            args=['-std=c++17'],
            unsaved_files=[(filename, code)]
        )
        return tu.cursor
    
    @staticmethod
    def ast_to_zss(node) -> Node:
        """Convert clang AST node to zss.Node for tree edit distance."""
        znode = Node(node.kind.name)
        for child in node.get_children():
            znode.addkid(CodeAnalyzer.ast_to_zss(child))
        return znode
    
    @staticmethod
    def count_zss_nodes(node: Node) -> int:
        """Count total number of nodes in a zss.Node tree."""
        return 1 + sum(CodeAnalyzer.count_zss_nodes(child) for child in node.children)
    
    def calculate_ast_distance(self, code1: str, code2: str) -> float:
        """Calculate normalized AST edit distance between two code snippets."""
        clean_code1 = self.strip_code_fence(code1)
        clean_code2 = self.strip_code_fence(code2)
        
        tree1 = self.ast_to_zss(self.parse_ast(clean_code1))
        tree2 = self.ast_to_zss(self.parse_ast(clean_code2))
        
        nodes1 = self.count_zss_nodes(tree1)
        nodes2 = self.count_zss_nodes(tree2)
        
        raw_distance = simple_distance(tree1, tree2)
        normalized_distance = raw_distance / (nodes1 + nodes2)
        
        return normalized_distance
    
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


class TestRunner:
    """Handles compilation and execution of generated code with unit tests."""
    
    @staticmethod
    def parse_unit_tests(test_text: str) -> List[Dict[str, str]]:
        """Parse unit test text into structured format."""
        block_pattern = r"(Unittest\s*\d+:\s*.*?)(?=Unittest\s*\d+:|$)"
        blocks = re.findall(block_pattern, test_text, flags=re.DOTALL)
        
        cases = []
        for block in blocks:
            match = re.search(r"Input:\s*(.*?)\s*Output:\s*(.*)", block, flags=re.DOTALL)
            if not match:
                continue
            
            unittest_num = re.match(r"Unittest\s*(\d+)", block).group(1)
            input_text = match.group(1).strip()
            output_text = match.group(2).strip()
            
            cases.append({
                "unittest": unittest_num,
                "input": input_text,
                "output": output_text
            })
        
        return cases
    
    @staticmethod
    def run_code_tests(code: str, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Compile and run code against test cases."""
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = Path(tmpdir) / "solution.cpp"
            bin_path = Path(tmpdir) / "solution"
            cpp_path.write_text(code)
            
            # Compile C++ code
            compile_result = subprocess.run(
                ["g++", str(cpp_path), "-std=c++17", "-O2", "-o", str(bin_path)],
                capture_output=True, text=True
            )
            
            if compile_result.returncode != 0:
                # Compilation failed
                for case in test_cases:
                    results.append({
                        "unittest": case["unittest"],
                        "passed": False,
                        "output": compile_result.stdout.strip(),
                        "error": compile_result.stderr.strip(),
                        "expected": case["output"]
                    })
                return results
            
            # Run tests
            for case in test_cases:
                proc = subprocess.run(
                    [str(bin_path)],
                    input=case["input"],
                    text=True,
                    capture_output=True
                )
                
                output = proc.stdout.strip()
                passed = (output == case["output"])
                
                results.append({
                    "unittest": case["unittest"],
                    "passed": passed,
                    "output": output,
                    "expected": case["output"]
                })
        
        return results


class PromptBuilder:
    """Builds prompts for code generation based on student history."""
    
    @staticmethod
    def build_student_prompt(student_df: pd.DataFrame, target_idx: int = 3) -> str:
        """Build prompt based on student's previous submissions."""
        if len(student_df) < target_idx + 1:
            raise ValueError(f"Student needs at least {target_idx + 1} submissions")
        
        examples = []
        for i in range(target_idx):
            q = student_df.iloc[i]
            examples.append(f"""Example {i+1}:
Question: {q['question_name']} — {q['question_text']}
Template:
{q['question_template']}
Your Code:
{q['response']}""")
        
        target_q = student_df.iloc[target_idx]
        
        prompt = f"""You are the same student who wrote the three examples below in your foundational C++ course. 
Mimic exactly your personal coding style, conventions, and level of proficiency—
do not over‐optimize or introduce unfamiliar patterns. 
Include the same sort of formatting, variable names, and minor imperfections you demonstrated. 
Respond ONLY with the C++ code (no commentary).

Week: {target_q['week']}
Topic: {target_q['topic']}

{chr(10).join(examples)}

Now, using that same student style, attempt this:
Question: {target_q['question_name']} — {target_q['question_text']}
Template:
{target_q['question_template']}

Provide ONLY your C++ implementation following the given template, 
writing code just as you would in class—indentation, naming, and all."""
        
        return prompt


class EvaluationPipeline:
    """Main evaluation pipeline that orchestrates the entire process."""
    
    def __init__(self):
        self.clients = LLMClients()
        self.generator = CodeGenerator(self.clients)
        self.analyzer = CodeAnalyzer()
        self.test_runner = TestRunner()
    
    def load_data(self) -> pd.DataFrame:
        """Load student submission data."""
        return pd.read_csv(Config.DATA_URL)
    
    def generate_code_for_students(self, df: pd.DataFrame, min_submissions: int = 4) -> Dict:
        """Generate code for all eligible students."""
        results = {}
        
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp").reset_index(drop=True)
            
            if len(student_df) < min_submissions:
                continue
            
            try:
                prompt = PromptBuilder.build_student_prompt(student_df)
                target_q = student_df.iloc[3]
                model_outputs = self.generator.generate_all(prompt)
                
                results[student_id] = {
                    "prompt": prompt,
                    "question_id": target_q['question_unittest_id'],
                    "ground_truth": target_q['response'],
                    "outputs": model_outputs
                }
            except Exception as e:
                print(f"Error processing student {student_id}: {e}")
                continue
        
        return results
    
    def evaluate_ast_distances(self, results: Dict) -> pd.DataFrame:
        """Evaluate AST edit distances for all generated codes."""
        all_rows = []
        
        for student_id, info in results.items():
            ground_truth = info["ground_truth"]
            
            for model_name, generated_code in info["outputs"].items():
                if "Error:" in generated_code:
                    normalized_distance = 1.0  # Maximum distance for errors
                else:
                    try:
                        normalized_distance = self.analyzer.calculate_ast_distance(
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
    
    def evaluate_embedding_similarities(self, results: Dict) -> pd.DataFrame:
        """Evaluate code embedding similarities for all generated codes."""
        all_rows = []
        
        for student_id, info in results.items():
            ground_truth = info["ground_truth"]
            
            for model_name, generated_code in info["outputs"].items():
                if "Error:" in generated_code:
                    similarity = 0.0
                else:
                    try:
                        similarity = self.analyzer.calculate_embedding_similarity(
                            generated_code, ground_truth
                        )
                    except Exception:
                        similarity = 0.0
                
                all_rows.append({
                    "student_id": student_id,
                    "question_id": info["question_id"],
                    "model": model_name,
                    "embedding_similarity": similarity
                })
        
        df_long = pd.DataFrame(all_rows)
        df_similarity = df_long.pivot(
            index=["student_id", "question_id"],
            columns="model",
            values="embedding_similarity"
        ).reset_index()
        
        return df_similarity
    
    def evaluate_unit_tests(self, results: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate unit test pass rates for all generated codes."""
        # Parse unit tests for each question
        test_cases_by_qid = {}
        for qid, group in df.groupby("question_unittest_id"):
            test_text = group["question_unittests"].iloc[0]
            test_cases_by_qid[qid] = self.test_runner.parse_unit_tests(test_text)
        
        all_rows = []
        
        for student_id, info in results.items():
            question_id = info["question_id"]
            test_cases = test_cases_by_qid.get(question_id, [])
            
            for model_name, generated_code in info["outputs"].items():
                if "Error:" in generated_code:
                    # Add failed results for all test cases
                    for case in test_cases:
                        all_rows.append({
                            "student_id": student_id,
                            "question_id": question_id,
                            "model": model_name,
                            "unittest": case["unittest"],
                            "passed": False,
                            "output": "Generation Error",
                            "expected": case["output"]
                        })
                else:
                    test_results = self.test_runner.run_code_tests(generated_code, test_cases)
                    for result in test_results:
                        all_rows.append({
                            "student_id": student_id,
                            "question_id": question_id,
                            "model": model_name,
                            **result
                        })
        
        return pd.DataFrame(all_rows)
    
    def run_full_evaluation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run complete evaluation pipeline."""
        print("Loading data...")
        df = self.load_data()
        
        print("Generating code for students...")
        results = self.generate_code_for_students(df)
        print(f"Generated code for {len(results)} students")
        
        print("Evaluating AST distances...")
        df_ast = self.evaluate_ast_distances(results)
        
        print("Evaluating embedding similarities...")
        df_similarity = self.evaluate_embedding_similarities(results)
        
        print("Evaluating unit tests...")
        df_tests = self.evaluate_unit_tests(results, df)
        
        return df_ast, df_similarity, df_tests
    
    def print_summary_metrics(self, df_ast: pd.DataFrame, df_similarity: pd.DataFrame, df_tests: pd.DataFrame):
        """Print summary evaluation metrics."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        print("\n1. AST Edit Distance (lower is better):")
        print("-" * 40)
        for model in ["Anthropic", "Gemini", "DeepSeek", "OpenAI"]:
            if model in df_ast.columns:
                avg_distance = df_ast[model].mean()
                print(f"{model:>12}: {avg_distance:.4f}")
        
        print("\n2. Code Embedding Similarity (higher is better):")
        print("-" * 50)
        for model in ["Anthropic", "Gemini", "DeepSeek", "OpenAI"]:
            if model in df_similarity.columns:
                avg_similarity = df_similarity[model].mean()
                print(f"{model:>12}: {avg_similarity:.4f}")
        
        print("\n3. Unit Test Pass Rate (higher is better):")
        print("-" * 42)
        if not df_tests.empty:
            pass_rates = df_tests.groupby('model')['passed'].mean()
            for model in pass_rates.index:
                print(f"{model:>12}: {pass_rates[model]:.4f}")


def main():
    """Main execution function."""
    pipeline = EvaluationPipeline()
    
    try:
        df_ast, df_similarity, df_tests = pipeline.run_full_evaluation()
        pipeline.print_summary_metrics(df_ast, df_similarity, df_tests)
        
        # Save results
        df_ast.to_csv("ast_distances.csv", index=False)
        df_similarity.to_csv("embedding_similarities.csv", index=False)
        df_tests.to_csv("unit_test_results.csv", index=False)
        
        print(f"\nResults saved to:")
        print("- ast_distances.csv")
        print("- embedding_similarities.csv") 
        print("- unit_test_results.csv")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()