import datasets
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists


class IFEvalScenario(Scenario):
    """IFEval

    IFEval contains around 500 "verifiable instructions" such as "write in more than 400 words"
    and "mention the keyword of AI at least 3 times" which can be verified by heuristics."""

    name = "ifeval"
    description = "Instruction-Following Evaluation for Large Language Models"
    tags = ["question answering"]

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get GPQA from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset("google/IFEval", trust_remote_code=True, cache_dir=cache_dir, split="train")
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        instances: List[Instance] = []
        for _, row in enumerate(dataset):
            input = Input(text=row["prompt"].strip())
            instance = Instance(
                input=input,
                references=[],
                split=TEST_SPLIT,
                extra_data={"instruction_id_list": row["instruction_id_list"], "question_kwargs": row["kwargs"]},
            )
            instances.append(instance)

        return instances
