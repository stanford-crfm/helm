from typing import List, Tuple, Dict, Any
import time

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.codeinsights_correct_code_metrics import (
    CodeInsightsFunctionalCorrectnessMetric,
    CPPEvaluator,
)


class CodeInsightsCodeEfficiencyMetric(CodeInsightsFunctionalCorrectnessMetric):
    """
    Comprehensive metric combining functional correctness and runtime efficiency evaluation.

    This metric first evaluates functional correctness and then measures runtime efficiency
    alignment between LLM-generated code and student reference code when both are correct.
    """

    def __init__(
        self,
        num_runtime_runs: int = 5,
        timeout_seconds: int = 10,
    ):
        """
        Initializes the CodeInsightsFunctionalCorrectnessMetric.

        Args:
            timeout (int): Timeout for each test case execution.
        """
        super().__init__()
        self.num_runtime_runs = num_runtime_runs
        self.timeout_seconds = timeout_seconds

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

        # Get the student code from the instance references
        student_code = request_state.instance.references[0].output.text.strip()
        print(f"Student code length: {len(student_code)}")

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
        evaluator = CPPEvaluator(
            question_template,
            test_cases,
            timeout=self.timeout_seconds,
            max_workers=1,
        )

        llm_output, llm_avg_runtime = self._timed_run(evaluator, generated_code, self.num_runtime_runs)
        stu_output, stu_avg_runtime = self._timed_run(evaluator, student_code, self.num_runtime_runs)

        # Compute functional correctness score
        if not llm_output or "score" not in llm_output:
            stats = [Stat(MetricName("functional_correctness")).add(0.0)]
        else:
            stats = [Stat(MetricName("functional_correctness")).add(llm_output["score"])]

        # Calculate runtime metrics if we have data for both solutions
        if llm_avg_runtime > 0 and stu_avg_runtime > 0:
            # Runtime ratio (LLM / Student) - values > 1 mean LLM is slower
            runtime_ratio = llm_avg_runtime / stu_avg_runtime if stu_avg_runtime > 0 else float("inf")

            # Efficiency alignment score (closer to 1.0 is better alignment)
            # Use reciprocal if LLM is faster to normalize the scale
            if runtime_ratio > 1:
                efficiency_alignment = 1.0 / runtime_ratio
            else:
                efficiency_alignment = runtime_ratio

            print(f"Runtime ratio (LLM/Student): {runtime_ratio:.4f}")
            print(f"Efficiency alignment score: {efficiency_alignment:.4f}")

            stats.extend(
                [
                    Stat(MetricName("runtime_efficiency_ratio")).add(runtime_ratio),
                    Stat(MetricName("efficiency_alignment_score")).add(efficiency_alignment),
                ]
            )

        # Handle cases where only one solution has runtime data
        elif llm_avg_runtime > 0 and stu_avg_runtime <= 0:
            print("Only LLM runtime available - student solution failed to run")
            stats.extend(
                [
                    Stat(MetricName("runtime_efficiency_ratio")).add(float("inf")),  # LLM runs, student doesn't
                    Stat(MetricName("efficiency_alignment_score")).add(0.0),  # No alignment possible
                ]
            )

        elif llm_avg_runtime <= 0 and stu_avg_runtime > 0:
            print("Only student runtime available - LLM solution failed to run")
            stats.extend(
                [
                    Stat(MetricName("runtime_efficiency_ratio")).add(0.0),  # Student runs, LLM doesn't
                    Stat(MetricName("efficiency_alignment_score")).add(0.0),  # No alignment possible
                ]
            )

        else:
            # Neither solution has runtime data
            print("Runtime measurement failed for both solutions")
            stats.extend(
                [
                    Stat(MetricName("runtime_efficiency_ratio")).add(0.0),
                    Stat(MetricName("efficiency_alignment_score")).add(0.0),
                ]
            )

        return stats

    def _timed_run(self, evaluator: CPPEvaluator, code: str, num_runtime_runs: int = 1) -> Tuple[Dict[str, Any], float]:
        list_runtimes: List[float] = []
        last_output: Dict[str, Any] = {}

        for _ in range(num_runtime_runs):
            start_time = time.perf_counter()
            output = evaluator.evaluate(code)
            passed = sum(output.get("testcases", []))

            if passed > 0:
                elapsed = time.perf_counter() - start_time
                list_runtimes.append(elapsed / passed)
                last_output = output
            # if passed == 0, we simply skip recording this run

        avg_runtime = sum(list_runtimes) / len(list_runtimes) if list_runtimes else 0.0
        return last_output, avg_runtime

    def _create_failure_stats(self, error_message: str) -> List[Stat]:
        """Create default statistics for failure cases."""
        print(f"RUNTIME EFFICIENCY METRIC FAILURE: {error_message}")
        return [
            Stat(MetricName("functional_correctness")).add(0.0),
            Stat(MetricName("runtime_efficiency_ratio")).add(0.0),
            Stat(MetricName("efficiency_alignment_score")).add(0.0),
        ]
