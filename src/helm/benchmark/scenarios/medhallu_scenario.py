from typing import Dict, List
from datasets import load_dataset

from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


class MedHalluScenario(Scenario):
    """
    MedHallu is a medical hallucination dataset that consists of PubMed articles and associated questions, with the objective being to classify whether the answer is factual or hallucinated.
    MedHallu: https://medhallu.github.io/
    """

    name = "medhallu"
    description = (
        "A dataset which consists of a PubMed article and a question "
    )
    tags = ["knowledge", "reasoning", "biomedical"]
    POSSIBLE_ANSWER_CHOICES: List[str] = ["1", "0"]

    def __init__(self):
        super().__init__()

    def process_csv(self, data, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Processing data for {split} split")
        for row in data:
            question = row["Question"]
            ground_truth_answer = row["Ground Truth Answer"]
            patient_note = row["Patient Note"]
            id = row["Row Number"]

            prompt = PassageQuestionInput(
                passage=patient_note + "\n", question=question + "\n", passage_prefix="Patient note: "
            )

            extra_data = {
                "category": row["Category"],
                "upper_limit": row["Upper Limit"],
                "lower_limit": row["Lower Limit"],
            }

            instance = Instance(
                input=prompt,
                references=[Reference(Output(text=ground_truth_answer), tags=[CORRECT_TAG])],
                extra_data=extra_data,
                split=split,
                id=id,
            )
            instances.append(instance)
        return instances

    def create_instance(self, question, knowledge, answer, label, split):
        prompt_text=f"""
            World Knowledge: {knowledge}

            Question: {question}

            Answer: {answer}


            Return just an integer value, '0' if the answer is factual and '1' if the answer is hallucinated. No letter or word, just the integer value.


            Your Judgment:
            """
        return Instance(
            input=PassageQuestionInput(
                passage="", question=prompt_text,
            ),
            references=[Reference(Output(text=label), tags=[CORRECT_TAG])],
            split=split,
        )
    def get_instances(self, output_path: str) -> List[Instance]:
        # Load the MedCalc-Bench dataset from Hugging Face
        dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", split="train")

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

            hallucinated_instance = self.create_instance(question, knowledge, hallucinated_answer, label="1", split=TEST_SPLIT)
            instances.append(hallucinated_instance)
        return instances
