import os
from typing import List
from common.general import ensure_file_downloaded
from benchmark.tokenizer.huggingface_tokenizer import HuggingFaceTokenizers

from .scenario import Scenario, Instance, Reference, TEST_SPLIT, CORRECT_TAG


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

    def __init__(self, num_input_tokens: int, num_instances: int, model: str):
        self.num_input_tokens: int = num_input_tokens
        self.num_instances: int = num_instances
        self.model = model

    def get_instances(self) -> List[Instance]:
        # Extract tokens from Alice in Wonderland
        data_path = os.path.join(self.output_path, "alice.txt")
        ensure_file_downloaded(
            source_url="https://www.gutenberg.org/files/11/11-0.txt", target_path=data_path, unpack=False,
        )
        with open(data_path, "r") as f:
            raw_text = f.read()
        text = raw_text.split(" ")
        batch_size = 256
        num_total_tokens = self.num_instances * self.num_input_tokens
        tokens: List[str] = []
        tokenizer = HuggingFaceTokenizers.get_tokenizer("huggingface/gpt2_tokenizer_fast")
        i = 0
        while len(tokens) < num_total_tokens:
            tokens += tokenizer.tokenize((" ").join(text[i * batch_size : (i + 1) * batch_size]))
            i += 1

        instances: List[Instance] = []
        for i in range(self.num_instances):
            per_instance_tokens = tokens[i * self.num_input_tokens : (i + 1) * self.num_input_tokens]
            per_instance_token_ids = tokenizer.convert_tokens_to_ids(per_instance_tokens)
            prompt = tokenizer.decode(per_instance_token_ids)
            instance = Instance(input=prompt, references=[Reference(output="", tags=[CORRECT_TAG])], split=TEST_SPLIT,)
            instances.append(instance)

        return instances
