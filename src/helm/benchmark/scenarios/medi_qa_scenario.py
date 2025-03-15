from typing import Dict, List
from datasets import load_dataset

from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class MediQAScenario(Scenario):
    """
    MEDIQA-QA is a dataset designed to benchmark large language models (LLMs) on medical
    question answering (QA) tasks.
    Each instance in the dataset includes a medical question, a set of candidate answers,
    relevance annotations for ranking, and additional context to evaluate understanding
    and retrieval capabilities in a healthcare setting.

    The dataset encompasses diverse question types, including consumer health queries
    and clinical questions, making it suitable for assessing LLMs' ability to answer
    consumer healthcare questions.

    This dataset comprises two training sets of 104 instances each, a validation set
    of 25 instances, and a testing set of 150 instances.

    Dataset: https://huggingface.co/datasets/bigbio/mediqa_qa
    Paper: https://aclanthology.org/W19-5039/

    Sample Prompt:
        Answer the following consumer health question.

        Question: Noonan syndrome. What are the references with noonan syndrome
        and polycystic renal disease?
        Answer:

    @inproceedings{MEDIQA2019,
        author    = {Asma {Ben Abacha} and Chaitanya Shivade and Dina Demner{-}Fushman},
        title     = {Overview of the MEDIQA 2019 Shared Task on Textual Inference,
                     Question Entailment and Question Answering},
        booktitle = {ACL-BioNLP 2019},
        year      = {2019}
    }
    """

    name = "medi_qa"
    description = (
        "A dataset including a medical question, a set of candidate answers,"
        "relevance annotations for ranking, and additional context to evaluate understanding"
        "and retrieval capabilities in a healthcare setting."
    )
    tags = ["knowledge", "biomedical"]

    def _get_highest_ranked_answer(self, answers: List[Dict[str, Dict[str, str]]]) -> str:
        best_answer: str = ""
        for answer in answers:
            if answer["Answer"]["ReferenceRank"] != 1:
                continue
            best_answer = answer["Answer"]["AnswerText"]
            break
        return best_answer

    def process_csv(self, data, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Processing data for {split} split")
        total_tokens: int = 0
        counter = 0
        for row in data:
            row = row["QUESTION"]
            question = row["QuestionText"]
            ground_truth_answer = self._get_highest_ranked_answer(row["AnswerList"])
            id = row["QID"]
            counter += 1
            total_tokens += len(ground_truth_answer.split())
            instances.append(
                Instance(
                    input=Input(question),
                    references=[Reference(Output(ground_truth_answer), tags=[CORRECT_TAG])],
                    split=split,
                    id=id,
                )
            )
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load the MEDIQA dataset from Hugging Face
        dataset = load_dataset("bigbio/mediqa_qa")

        # Process all the instances
        instances: List[Instance] = []
        # Limit to zero shot setting
        splits: Dict[str, str] = {
            # "train_live_qa_med": TRAIN_SPLIT,
            # "validation": VALID_SPLIT,
            "test": TEST_SPLIT,
        }
        for hf_split, split in splits.items():
            data = dataset[hf_split]
            instances.extend(self.process_csv(data, split))

        return instances
