import json
import os
from typing import Dict, List

from datasets import DatasetDict, load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_directory_exists


class MedCalcBenchScenario(Scenario):
    """
    MedCalcBench scenario: Processes a medical calculation dataset with explanations.

    Each record in the dataset has:
    - Row Number
    - Calculator ID
    - Calculator Name
    - Category
    - Output Type
    - Note ID
    - Note Type
    - Question
    - Ground Truth Explanation
    - Patient Note
    - Relevant Entities
    - Lower Limit
    - Upper Limit
    - Ground Truth Answer

    The output is formatted as:
    "The answer is <calculated value>. Steps: <explanation>"
    """

    HUGGING_FACE_DATASET_PATH: str = "ncbi/MedCalc-Bench-v1.0"

    # TODO: Add a base url
    DATASET_DOWNLOAD_BASE_URL: str = ""

    name = "medcalcbench"
    description = "Medical calculation questions with step-by-step explanations."
    tags = ["reasoning", "medicine", "calculation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(self.HUGGING_FACE_DATASET_PATH)

        splits = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}
        instances: List[Instance] = []
        for (
            helm_split_name,
            dataset_split_name,
        ) in splits.items():  # Iterate over the splits
            split_data = dataset[dataset_split_name]

            for example in split_data:
                question = example["Question"]
                patient_note = example["Patient Note"]

                input_text = (
                    f"Patient Note:\n\n{patient_note}\n\nQuestion:\n\n{question}"
                )

                # Format the final answer with explanation
                instances.append(
                    Instance(
                        input=Input(text=input_text),
                        references=[
                            Reference(
                                Output(text=example["Ground Truth Answer"]),
                                tags=[CORRECT_TAG],
                            )
                        ],
                        split=helm_split_name,
                        extra_data={
                            "id": example["Row Number"],
                            "relevant_entities": example["Relevant Entities"],
                            "lower_limit": example["Lower Limit"],
                            "upper_limit": example["Upper Limit"],
                            "calculator_id": example["Calculator ID"],
                            "ground_truth": example["Ground Truth Answer"],
                        },
                    )
                )

        return instances
