import os
from typing import List
from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, TEST_TAG


class SyntheticScenario(Scenario):
    """
    This synthetic scenario is intended for conducting efficiency-oriented
    benchmarking. In particular, we seek to address the following questions:

    1. What is the dependence of runtime on number of tokens in the prompt and
       number of generated output tokens? How about number of completions?
    2. How much variance do we observe for each query?
    3. What's the difference between warm start and cold start inference times?
    4. How do different models (across providers) behave?
    5. Can we reverse engineer the hardware used by providers?

    We gather input text from a fixed public domain source (in particular,
    Alice in Wonderland) and vary various parameters, including the model,
    the number of input and output tokens, the number of input instances,
    the number of output completions.
    """

    name = "synthetic"
    description = "Synthetic scenario for benchmarking efficiency metrics"
    tags = []

    def __init__(self, num_tokens_per_instance: int, num_instances: int):
        self.num_instances = num_instances
        self.num_tokens_per_instance = num_tokens_per_instance
        self.num_input_tokens = self.num_instances * self.num_tokens_per_instance

    def get_instances(self) -> List[Instance]:
        # Extract tokens from Alice in Wonderland
        data_path = os.path.join(self.output_path, "alice.txt")
        ensure_file_downloaded(
            source_url="https://www.gutenberg.org/files/11/11-0.txt",
            target_path=data_path,
            unpack=False,
        )
        tokens = []
        with open(data_path, "r") as f:
            for line in f:
                tokens += line.strip().split(" ")
                if len(tokens) >= self.num_input_tokens:
                    break

        instances: List[Instance] = []
        for i in range(self.num_instances):
            assert len(tokens) >= self.num_tokens_per_instance
            instance = Instance(
                input=" ".join(tokens[: self.num_tokens_per_instance]),
                references=[Reference(output="", tags=[])],
                tags=[TEST_TAG],
            )
            instances.append(instance)
            tokens = tokens[self.num_tokens_per_instance :]

        return instances
