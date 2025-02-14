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
from helm.common.general import ensure_file_downloaded, ensure_directory_exists


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
    ONE_SHOT_EXAMPLES_URL = "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/one_shot_finalized_explanation.json"

    name = "medcalcbench"
    description = "Medical calculation questions with step-by-step explanations."
    tags = ["reasoning", "medicine", "calculation"]

    def __init__(self, is_one_shot: bool):
        super().__init__()
        self.is_one_shot = is_one_shot

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(self.HUGGING_FACE_DATASET_PATH)

        if self.is_one_shot:
            one_shot_examples_path = os.path.join(output_path, "one_shot_examples")
            ensure_file_downloaded(
                source_url=self.ONE_SHOT_EXAMPLES_URL, target_path=one_shot_examples_path
            )

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
                if self.is_one_shot:
                    one_shot_examples = json.load(open(one_shot_examples_path))

                    # In one-shot, we have a single example for each calculator ID
                    calculator_id = str(example["Calculator ID"])
                    instructions = self._get_one_shot_cot_instructions(
                        question, calculator_id, one_shot_examples
                    )

                    input_text = f"{instructions}\n\n{input_text}"

                # Format the final answer with explanation
                instances.append(
                    Instance(
                        id=example["Row Number"],
                        input=Input(text=input_text),
                        references=[
                            Reference(
                                Output(text=example["Ground Truth Answer"]),
                                tags=[CORRECT_TAG],
                            )
                        ],
                        split=helm_split_name,
                        extra_data={
                            "lower_limit": example["Lower Limit"],
                            "upper_limit": example["Upper Limit"],
                            "calculator_id": example["Calculator ID"],
                        },
                    )
                )

        return instances

    def _get_one_shot_cot_instructions(self, question: str, calculator_id: str, one_shot_examples: dict) -> str:
        """Generate instructions for the MedCalc-Bench scenario.

        This function is inspired on the system prompt definition in the original code:
        https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/run.py#L26

        Credits to the original authors: https://github.com/ncbi-nlp/MedCalc-Bench.

        In the original code, there's exactly one example response for each calculator ID.
        These examples are stored in a JSON file: https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/one_shot_finalized_explanation.json
        None of the examples include the actual questions. They only contain the step-by-step thinking and the final answer.
        Looking at the dataset samples we can see that all samples with the same calculator ID use the same question.
        The original expect that for each sample, we collect the calculator ID and the question for building the one-shot instructions.
        """
        if not one_shot_examples:
            raise ValueError(
                "Failed to load one-shot examples for the MedCalc-Bench scenario."
            )

        example = one_shot_examples.get(calculator_id, {})

        if not example:
            raise ValueError(
                f"Failed to find one-shot example for calculator ID {calculator_id}."
            )

        return (
            "Below is an example:"
            # This example follows the formatting of the respective scenario.
            f"\n\nPatient Note:\n\n{example['Patient Note']}"
            f"\n\nQuestion:\n\n{question}"
            f"\n\nExplanation:\n\n{example['Response']['step_by_step_thinking']}"
            f"\n\nCalculated Value: {example['Response']['answer']}"
        )
