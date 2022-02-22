import os
from typing import List, Union
from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_TAG


class SyntheticScenario(Scenario):

    name = "synthetic"
    description = "Synthetic scenario for benchmarking efficiency metrics"
    tags = []

    def __init__(self, num_tokens_per_instance: Union[int, List[int]], num_instances: int):
        self.num_instances = num_instances
        if isinstance(num_tokens_per_instance, list):
            assert (
                len(num_tokens_per_instance) == self.num_instances,
                f"Requested {self.num_instances} instances but "
                f"{len(num_tokens_per_instance)} instance lengths specified",
            )
            self.num_tokens_per_instance = num_tokens_per_instance
        else:
            self.num_tokens_per_instance = [num_tokens_per_instance] * num_instances
        self.num_input_tokens = sum(self.num_tokens_per_instance)

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
            assert len(tokens) >= self.num_tokens_per_instance[i]
            instance = Instance(
                input=" ".join(tokens[: self.num_tokens_per_instance[i]]),
                references=[Reference(output="", tags=[])],
                tags=[TEST_TAG],
            )
            instances.append(instance)
            tokens = tokens[self.num_tokens_per_instance[i] :]

        return instances
