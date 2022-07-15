import os
from typing import List
import urllib.parse

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, TEST_SPLIT, CORRECT_TAG

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

BASE_URL = (
    "https://raw.githubusercontent.com/stanford-crfm/benchmarking_efficiency/main/synthetic_efficiency_instances/"
)


class SyntheticEfficiencyScenario(Scenario):
    """
    This synthetic scenario is intended for conducting efficiency-oriented
    benchmarking. In particular, we seek to address the following questions:

    1. What is the dependence of runtime on number of tokens in the prompt and
       number of generated output tokens? How about number of completions?
    2. How much variance do we observe for each query?
    3. What's the difference between warm start and cold start inference times?
    4. How do different models (across providers) behave?
    5. Can we reverse engineer the hardware used by providers?

    We gather input text from fixed public domain sources and vary various parameters,
    including the model the number of input and output tokens, the number of
    input instances, the number of output completions.
    """

    name = "synthetic_efficiency"
    description = "Synthetic scenario for benchmarking efficiency metrics"
    tags: List[str] = []

    def __init__(self, num_input_tokens: int, num_instances: int, tokenizer: str):
        self.num_input_tokens: int = num_input_tokens
        self.num_instances: int = num_instances
        self.tokenizer = tokenizer
        assert self.tokenizer in ["huggingface/gpt2_tokenizer_fast", "ai21"]

    def get_instances(self) -> List[Instance]:
        data_paths = []
        for instance_id in range(self.num_instances):
            parameters = (
                f"input_tokens={self.num_input_tokens},"
                f"tokenizer={self.tokenizer.replace('/', '_')},"
                f"id={instance_id}.txt"
            )
            source_url = f"{BASE_URL}{urllib.parse.quote(parameters)}"
            data_path = os.path.join(self.output_path, parameters.replace("/", "_"))
            ensure_file_downloaded(
                source_url=source_url, target_path=data_path, unpack=False,
            )
            data_paths.append(data_path)

        instances = []
        for data_path in data_paths:
            with open(data_path, "r") as f:
                prompt = f.read()
                instance = Instance(
                    input=prompt, references=[Reference(output="", tags=[CORRECT_TAG])], split=TEST_SPLIT,
                )
                instances.append(instance)
        return instances
