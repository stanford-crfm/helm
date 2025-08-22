from typing import List
from datasets import load_dataset

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Output,
    Input,
    ScenarioMetadata,
)


class MedHalluScenario(Scenario):
    """
    MedHallu is a medical hallucination dataset that consists of PubMed articles and associated questions,
    with the objective being to classify whether the answer is factual or hallucinated.
    MedHallu: https://medhallu.github.io/
    """

    name = "medhallu"
    description = (
        "MedHallu is a benchmark focused on evaluating factual correctness in biomedical"
        "question answering. Each instance contains a PubMed-derived knowledge snippet, a"
        "biomedical question, and a model-generated answer. The task is to classify whether the"
        "answer is factually correct or contains hallucinated (non-grounded) information. This"
        "benchmark is designed to assess the factual reliability of medical language models."
    )
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

    def get_metadata(self):
        return ScenarioMetadata(
            name="medhallu",
            display_name="MedHallu",
            description="MedHallu is a benchmark focused on evaluating factual correctness in "
            "biomedical question answering. Each instance contains a PubMed-derived "
            "knowledge snippet, a biomedical question, and a model-generated answer. The "
            "task is to classify whether the answer is factually correct or contains "
            "hallucinated (non-grounded) information. This benchmark is designed to assess "
            "the factual reliability of medical language models.",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Verify whether answers to questions from PubMed articles are " "factual or hallucinated",
                when="Any",
                who="Researcher",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
