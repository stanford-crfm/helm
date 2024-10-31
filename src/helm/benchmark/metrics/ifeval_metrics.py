from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

import src.helm.benchmark.metrics.ifeval_instructions_registry as instructions_registry


class IFEvalMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        prompt = request_state.instance.input.text
        instruction_id_list = request_state.instance.extra_data["instruction_id_list"]
        question_kwargs = request_state.instance.extra_data["question_kwargs"]
        assert len(instruction_id_list) > 0
        assert request_state.result
        assert len(request_state.result.completions) == 1, len(request_state.result.completions)
        response = request_state.result.completions[0].text.strip()

        is_following_list = []
        for index, instruction_id in enumerate(instruction_id_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**{k: v for k, v in question_kwargs[index].items() if v is not None})
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            if response.strip() and instruction.check_following(response):
                is_following_list.append(1)
            else:
                is_following_list.append(0)

        return [Stat(MetricName("strict_accuracy")).add(sum(is_following_list) / len(is_following_list))]
