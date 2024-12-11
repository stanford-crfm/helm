import os
from typing import List, Dict
import json
from helm.common.general import ensure_file_downloaded
from helm.common.constants import TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, Input, Output

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

    # TODO: Add a base url
    DATASET_DOWNLOAD_BASE_URL: str = ""

    name = "medcalcbench"
    description = "Medical calculation questions with step-by-step explanations."
    tags = ["reasoning", "medicine", "calculation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        instances: List[Instance] = []

        for split, split_tag in splits.items():  # Iterate over the splits
            source_url: str = f"{self.DATASET_DOWNLOAD_BASE_URL}/{split}.jsonl"
            data_path: str = os.path.join(output_path, f"med_calc_bench_{split}")
            ensure_file_downloaded(source_url=source_url, target_path=data_path)

            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    example: Dict = json.loads(line.strip())
                    question = example["Question"]
                    patient_note = example["Patient_Note"]

                    input_text = f"Patient Note:\n\n{patient_note}\n\nQuestion:\n\n{question}"

                    # Format the final answer with explanation
                    instances.append(
                        Instance(
                            input=Input(text=input_text),
                            references=[Reference(Output(text=example["Ground_Truth_Answer"]), tags=[CORRECT_TAG])],
                            split=split_tag,
                            extra_data={
                                "relevant_entities": example["Relevant_Entities"],
                                "lower_limit": example["Lower_Limit"],
                                "upper_limit": example["Upper_Limit"],
                                "calculator_id": example["Calculator ID"],
                                "ground_truth": example["Ground_Truth_Answer"],
                            }
                        )
                    )

        return instances