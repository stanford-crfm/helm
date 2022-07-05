import os
from typing import Dict, List
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
    sources = {
        "alice": "https://www.gutenberg.org/files/11/11-0.txt",
        "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "monte_cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
        "crime_and_punishment": "https://www.gutenberg.org/files/2554/2554-0.txt",
    }

    def __init__(self, num_input_tokens: int, num_instances: int):
        self.num_input_tokens: int = num_input_tokens
        self.num_instances: int = num_instances
        assert self.num_instances % len(self.sources) == 0

    def get_instances(self) -> List[Instance]:
        books: List[str] = list(self.sources.keys())
        tokens: Dict = {}
        tokenizer = HuggingFaceTokenizers.get_tokenizer("huggingface/gpt2_tokenizer_fast")
        # Extract tokens from book sources
        for book, source_url in self.sources.items():
            data_path = os.path.join(self.output_path, f"{book}.txt")
            ensure_file_downloaded(
                source_url=source_url, target_path=data_path, unpack=False,
            )
            with open(data_path, "r") as f:
                text = f.read()
            batch_size = 2048
            # Skip intro text
            text = text[batch_size:]
            num_total_tokens_per_book = (self.num_instances * self.num_input_tokens) // len(books)
            i = 0
            tokens[book] = []
            while len(tokens[book]) < num_total_tokens_per_book:
                tokens[book] += tokenizer.tokenize(text[i * batch_size : (i + 1) * batch_size])
                i += 1

        instances: List[Instance] = []
        for i in range(self.num_instances // len(books)):
            for j in range(len(books)):
                per_instance_tokens = tokens[books[j]][i * self.num_input_tokens : (i + 1) * self.num_input_tokens]
                assert len(per_instance_tokens) == self.num_input_tokens
                per_instance_token_ids = tokenizer.convert_tokens_to_ids(per_instance_tokens)
                prompt = tokenizer.decode(per_instance_token_ids)
                instance = Instance(
                    input=prompt, references=[Reference(output="", tags=[CORRECT_TAG])], split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances
