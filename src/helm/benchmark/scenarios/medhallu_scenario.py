from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Output,
    Input,
)


class MedHalluScenario(Scenario):
    """
    MedHallu is a medical hallucination dataset that consists of PubMed articles and associated questions,
    with the objective being to classify whether the answer is factual or hallucinated.
    MedHallu: https://medhallu.github.io/
    """

    name = "medhallu"
    description = "A dataset of PubMed articles and associated questions, with the objective being to classify whether the answer is factual or hallucinated."  # noqa: E501
    tags = ["knowledge", "reasoning", "biomedical"]

    def create_instance(self, question, knowledge, answer, label, split):
        prompt_text = f"""World Knowledge: {knowledge}

Question: {question}

Answer: {answer}
"""
        return Instance(
            input=Input(
                text=prompt_text,
            ),
            references=[Reference(Output(text=label), tags=[CORRECT_TAG])],
            split=split,
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load the MedCalc-Bench dataset from Hugging Face
        dataset = load_dataset(
            "UTAustin-AIHealth/MedHallu",
            "pqa_labeled",
            split="train",
            revision="515060458a945c633debc6fd5baac7764416b724",
        )

        # Process all the instances - limit to zero shot setting
        instances: List[Instance] = []
        for row in dataset:
            # print("printing row", row)
            question = row["Question"]
            ground_truth_answer = row["Ground Truth"]
            knowledge = row["Knowledge"]
            hallucinated_answer = row["Hallucinated Answer"]

            gt_instance = self.create_instance(question, knowledge, ground_truth_answer, label="0", split=TEST_SPLIT)
            instances.append(gt_instance)

            hallucinated_instance = self.create_instance(
                question, knowledge, hallucinated_answer, label="1", split=TEST_SPLIT
            )
            instances.append(hallucinated_instance)
        return instances
