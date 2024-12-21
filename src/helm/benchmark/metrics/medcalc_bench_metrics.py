import re
from datetime import datetime
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class MedCalcBenchMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Metric for MedCalc-Bench dataset.

        Original implementation:
        https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/evaluate.py#L11
        """
        assert request_state.instance.extra_data, (
            "Could not find `extra_data` in the request state. "
            "Both `lower_limit` and `upper_limit` are required for this metric."
        )

        assert len(request_state.result.completions) == 1, (
            f"Found a total of {len(request_state.result.completions)} completions. "
            "Only one was expected"
        )

        final_answer = (
            request_state.result.completions[0]
            .text.strip()
            .lower()
            .split("calculated value:")[-1]
            .strip()
        )

        correctness = 0
        if final_answer:
            try:
                correctness = self.medcalc_bench_metric_calculation(
                    answer=final_answer,
                    ground_truth=request_state.instance.extra_data["ground_truth"],
                    calid=int(request_state.instance.extra_data["calculator_id"]),
                    upper_limit=request_state.instance.extra_data["upper_limit"],
                    lower_limit=request_state.instance.extra_data["lower_limit"],
                )
            except ValueError as e:
                hlog(
                    (
                        "Failed to calculate the correctess of the output for MedCalc-Bench instance "
                        f'with id {request_state.instance.extra_data["id"]}: {e}'
                    )
                )

        return [Stat(MetricName("medcalc_bench_metric")).add(correctness)]

    def medcalc_bench_metric_calculation(
        self,
        answer: str,
        ground_truth: str,
        calid: int,
        upper_limit: str,
        lower_limit: str,
    ) -> int:
        """Calculate the metric for MedCalc-Bench dataset.

        This method is basically a copy of the original implementation of this metric:
        https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/evaluate.py#L11

        Credits to the original authors: https://github.com/ncbi-nlp/MedCalc-Bench.
        """
        if calid in [13, 68]:
            # Output Type: date

            if datetime.strptime(answer, "%m/%d/%Y").strftime(
                "%-m/%-d/%Y"
            ) == datetime.strptime(ground_truth, "%m/%d/%Y").strftime("%-m/%-d/%Y"):
                correctness = 1
            else:
                correctness = 0
        elif calid in [69]:
            # Output Type: integer (A, B)
            match = re.search(
                r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?",
                ground_truth,
            )
            ground_truth = f"({match.group(1)}, {match.group(3)})"
            match = re.search(
                r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?",
                answer,
            )
            if match:
                weeks = match.group(1)
                days = match.group(3)
                answer = f"({weeks}, {days})"
                if eval(answer) == eval(ground_truth):
                    correctness = 1
                else:
                    correctness = 0
            else:
                correctness = 0
        elif calid in [
            4,
            15,
            16,
            17,
            18,
            20,
            21,
            25,
            27,
            28,
            29,
            32,
            33,
            36,
            43,
            45,
            48,
            51,
            69,
        ]:
            # Output Type: integer A
            answer = round(int(answer))
            if answer == int(ground_truth):
                correctness = 1
            else:
                correctness = 0
        elif calid in [
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            19,
            22,
            23,
            24,
            26,
            30,
            31,
            38,
            39,
            40,
            44,
            46,
            49,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
        ]:
            # Output Type: decimal
            answer = float(answer)
            if answer >= float(lower_limit) and answer <= float(upper_limit):
                correctness = 1
            else:
                correctness = 0
        else:
            raise ValueError(f"Unknown calculator ID: {calid}")
        return correctness
