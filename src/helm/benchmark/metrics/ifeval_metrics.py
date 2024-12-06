from typing import List

from helm.common.hierarchical_logger import hlog
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

from helm.benchmark.metrics.ifeval.instructions_registry import INSTRUCTION_DICT


class IFEvalMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        prompt = request_state.instance.input.text
        assert request_state.instance.extra_data
        instruction_ids = request_state.instance.extra_data["instruction_ids"]
        instruction_kwargs = request_state.instance.extra_data["instruction_kwargs"]
        assert len(instruction_ids) > 0
        assert request_state.result
        assert len(request_state.result.completions) == 1, f"Got {len(request_state.result.completions)} completions"
        response = request_state.result.completions[0].text.strip()

        # The following logic was reproduced with minor modifications from the following URL:
        # https://github.com/google-research/google-research/blob/c7f60c013623e613732a096e2a0c2872491ec912/
        # instruction_following_eval/evaluation_main.py#L96-L125
        is_following_list = []
        for index, instruction_id in enumerate(instruction_ids):
            instruction_cls = INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**{k: v for k, v in instruction_kwargs[index].items() if v is not None})
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            is_following = False
            if response.strip():
                try:
                    is_following = instruction.check_following(response)
                except Exception as e:
                    hlog(f"WARNING: Instruction following checking failed with error message {e}")
            if is_following:
                is_following_list.append(1)
            else:
                is_following_list.append(0)

        return [Stat(MetricName("ifeval_strict_accuracy")).add(sum(is_following_list) / len(is_following_list))]
