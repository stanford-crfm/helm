import os
from typing import List, Optional

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.ruler_qa_scenario_helper import generate_samples  # type: ignore
from helm.benchmark.scenarios.scenario import (
    VALID_SPLIT,
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)


_DATASET_TO_URL = {
    "hotpotqa": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "squad": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}


class RulerQAScenario(Scenario):
    name = "ruler_qa"
    description = "A QA scenario from Ruler"
    tags = ["long_context", "rag"]

    _TEMPLATE = """Answer the question based on the given documents. Only give me the answer and do not output any other words.

The following are given documents.

{context}

Answer the question based on the given documents. Only give me the answer and do not output any other words.

Question: {query} Answer:"""  # noqa: E501

    def __init__(self, dataset: Optional[str] = None, max_sequence_length: Optional[int] = None):
        super().__init__()
        self.dataset = dataset or "hotpotqa"
        self.max_sequence_length = max_sequence_length or 32768

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)
        file_path = os.path.join(data_dir, f"{self.dataset}.json")
        url = _DATASET_TO_URL[self.dataset]
        ensure_file_downloaded(url, file_path)
        instances: List[Instance] = []
        samples = generate_samples(
            dataset=self.dataset,
            dataset_path=file_path,
            max_seq_length=self.max_sequence_length,
            tokens_to_generate=32,
            num_samples=500,
            random_seed=42,
            dataset=self.dataset,
            pre_samples=0,
            template=self._TEMPLATE,
        )
        for sample in samples:
            instance = Instance(
                id=sample["index"],
                input=Input(text=sample["input"]),
                references=[
                    Reference(Output(text=output_text), tags=[CORRECT_TAG]) for output_text in sample["outputs"]
                ],
                split=VALID_SPLIT,
            )
            instances.append(instance)
        return instances
