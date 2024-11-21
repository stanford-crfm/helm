import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, TEST_SPLIT, CORRECT_TAG, Input, Output

NUM_INPUT_TOKENS: List[int] = [
    1,
    16,
    32,
    64,
    256,
    512,
    768,
    1024,
    1536,
    1920,
]
NUM_OUTPUT_TOKENS: List[int] = [1, 2, 4, 8, 16, 32, 64]


class SyntheticEfficiencyScenario(Scenario):
    """
    This synthetic scenario is intended for conducting efficiency-oriented
    benchmarking. In particular, we seek to address the following questions:

    1. What is the dependence of runtime on number of tokens in the prompt and
       number of generated output tokens? How about number of completions?
    2. How much variance do we observe for each query?
    3. How do different models (across providers) behave?
    4. Can we reverse engineer the hardware used by providers?

    We gather input text from fixed public domain sources and vary various parameters,
    including the model the number of input and output tokens, the number of
    input instances, the number of output completions.

    The dataset is stored at https://worksheets.codalab.org/bundles/0x17a361bc066b4b0e87d968069759d361.
    """

    name = "synthetic_efficiency"
    description = "Synthetic scenario for benchmarking efficiency metrics"
    tags: List[str] = []

    def __init__(self, num_prompt_tokens: int, num_instances: int, tokenizer: str):
        super().__init__()
        self.num_prompt_tokens: int = num_prompt_tokens
        self.num_instances: int = num_instances
        self.tokenizer: str = tokenizer

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        assert self.tokenizer in [
            "huggingface/gpt2",
            "ai21/j1",
            "cohere/cohere",
            "meta/opt",
            "yandex/yalm",
            "bigscience/bloom",
            "bigscience/t0pp",
            "google/t5",
            "google/ul2",
            "tsinghua/glm",
            "eleutherai/gptj",
            "eleutherai/gptneox",
        ]
        for instance_id in range(self.num_instances):
            file_name: str = (
                f"num_prompt_tokens={self.num_prompt_tokens},"
                f"tokenizer={self.tokenizer.replace('/', '_')},"
                f"id={instance_id}.txt"
            )
            data_path: str = os.path.join(output_path, file_name)
            ensure_file_downloaded(
                source_url=f"https://worksheets.codalab.org/rest/bundles/0x17a361bc066b4b0e87d968069759d361/"
                f"contents/blob/{file_name}",
                target_path=data_path,
                unpack=False,
            )

            with open(data_path, "r") as f:
                prompt: str = f.read()
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=""), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances
