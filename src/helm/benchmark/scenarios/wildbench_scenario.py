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
            trust_remote_code=True,
            cache_dir=cache_dir,
            split="test",
        )
        assert isinstance(dataset, datasets.Dataset)
        baseline_outputs = {
            f"{model}": datasets.load_dataset(
                "allenai/WildBench-V2-Model-Outputs",
                model,
                trust_remote_code=True,
                cache_dir=cache_dir,
                split="train",
            )
            for model in REFERENCE_MODELS
        }
        assert all(isinstance(baseline_output, datasets.Dataset) for baseline_output in baseline_outputs.values())

        # Read all instances
        instances: List[Instance] = []
        for idx, row in enumerate(dataset):
            input_text = []
            history_text = []
            for round in row["conversation_input"][:-1]:
                noun = "User: " if round["role"] == "user" else "Assistant: "
                history_text.append(noun + round["content"])
                input_text.append(noun + round["content"])

            round = row["conversation_input"][-1]
            noun = "User: "
            input_text.append(noun + round["content"])
            user_query = round["content"]

            input = Input(text="\n".join(input_text))
            instance = Instance(
                input=input,
                references=[],
                split=TEST_SPLIT,
                extra_data={
                    "baseline_outputs": {
                        model: baseline_outputs[model][idx]["output"][0] for model in REFERENCE_MODELS
                    },
                    "history": "\n".join(history_text),
                    "conversation": row["conversation_input"],
                    "user_query": user_query,
                    "checklist": "\n".join(row["checklist"]),
                },
            )
            instances.append(instance)

        return instances
