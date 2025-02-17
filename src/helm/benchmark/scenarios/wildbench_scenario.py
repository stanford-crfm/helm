import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)
from helm.common.general import ensure_directory_exists


SUBSETS = ["v2"]
REFERENCE_MODELS = ["gpt-4-turbo-2024-04-09", "claude-3-haiku-20240307", "Llama-2-70b-chat-hf"]


class WildBenchScenario(Scenario):
    """WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild

    WildBench is a benchmark for evaluating large language models (LLMs) on challenging tasks
    that are more representative of real-world applications. The examples are collected from
    real users by the AI2 WildChat project."""

    name = "wildbench"
    description = "Benchmarking LLMs with Challenging Tasks from Real Users in the Wild"
    tags = ["instruction following"]

    def __init__(self, subset: str, use_model_outputs: bool = False):
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset
        self.use_model_outputs = use_model_outputs

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get WildBench from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "allenai/WildBench",
            self.subset,
            cache_dir=cache_dir,
            split="test",
            revision="7c05c1b4550282b2ed6a2e6ac5db069f1e07df5c",
        )
        assert isinstance(dataset, datasets.Dataset)
        if self.use_model_outputs:
            baseline_outputs = {
                f"{model}": datasets.load_dataset(
                    "allenai/WildBench-V2-Model-Outputs",
                    model,
                    cache_dir=cache_dir,
                    split="train",
                    revision="d6755bc68220df853c0825a733430f73f5af2501",
                )
                for model in REFERENCE_MODELS
            }
            assert all(isinstance(baseline_output, datasets.Dataset) for baseline_output in baseline_outputs.values())

        # Read all instances
        instances: List[Instance] = []
        for idx, row in enumerate(dataset):
            input = Input(
                messages=[
                    {"role": message["role"], "content": message["content"]} for message in row["conversation_input"]
                ]
            )
            extra_data = {
                "checklist": row["checklist"],
            }
            if self.use_model_outputs:
                extra_data["baseline_outputs"] = {
                    model: baseline_outputs[model][idx]["output"][0] for model in REFERENCE_MODELS
                }
            instance = Instance(
                input=input,
                references=[],
                split=TEST_SPLIT,
                extra_data=extra_data,
            )
            instances.append(instance)

        return instances
