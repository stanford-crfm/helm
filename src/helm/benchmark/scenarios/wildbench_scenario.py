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

    def __init__(self, subset: str):
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get WildBench from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "allenai/WildBench",
            self.subset,
            cache_dir=cache_dir,
            split="test",
        )
        assert isinstance(dataset, datasets.Dataset)
        baseline_outputs = {
            f"{model}": datasets.load_dataset(
                "allenai/WildBench-V2-Model-Outputs",
                model,
                cache_dir=cache_dir,
                split="train",
            )
            for model in REFERENCE_MODELS
        }
        assert all(isinstance(baseline_output, datasets.Dataset) for baseline_output in baseline_outputs.values())

        # Read all instances
        instances: List[Instance] = []
        for idx, row in enumerate(dataset):

            conversation = row["conversation_input"]

            # Following https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/src/eval.py
            history = []
            for round in row["conversation_input"][:-1]:
                noun = "USER: " if round["role"] == "user" else "ASSISTANT: "
                history.append(noun + round["content"])
            history_text = "\n\n".join(history)
            user_query_text = row["conversation_input"][-1]["content"]
            checklist_text = "\n".join([f"- {checklist_item}" for checklist_item in row["checklist"]])

            input = Input(text="")  # placeholder, adapter actually uses the conversation for Request messages
            instance = Instance(
                input=input,
                references=[],
                split=TEST_SPLIT,
                extra_data={
                    "conversation": conversation,
                    "baseline_outputs": {
                        model: baseline_outputs[model][idx]["output"][0] for model in REFERENCE_MODELS
                    },
                    "history": history_text,
                    "user_query": user_query_text,
                    "checklist": checklist_text,
                },
            )
            instances.append(instance)

        return instances
