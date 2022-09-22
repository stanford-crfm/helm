import json
import os
import random
from typing import Dict, List

from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, TEST_SPLIT


class EmrQAScenario(Scenario):
    """
    From "emrQA: A Large Corpus for Question Answering on Electronic Medical Records", emrQA
    is an electronic medical record QA dataset that contains patient-specific information.

    To generate the emrQA dataset, you would first need to sign an agreement and
    download the i2b2 datasets. Follow the instructions here:
    https://github.com/panushri25/emrQA#download-dataset.
    Then place the output files under `restricted/emrQA/dataset`.

    Example from the dataset:


    @inproceedings{pampari-etal-2018-emrqa,
        title = "emr{QA}: A Large Corpus for Question Answering on Electronic Medical Records",
        author = "Pampari, Anusri  and
          Raghavan, Preethi  and
          Liang, Jennifer  and
          Peng, Jian",
        booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
        month = oct # "-" # nov,
        year = "2018",
        address = "Brussels, Belgium",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/D18-1258",
        doi = "10.18653/v1/D18-1258",
        pages = "2357--2368",
        abstract = "We propose a novel methodology to generate domain-specific large-scale question answering (QA)
        datasets by re-purposing existing annotations for other NLP tasks. We demonstrate an instance of this
        methodology in generating a large-scale QA dataset for electronic medical records by leveraging existing
        expert annotations on clinical notes for various NLP tasks from the community shared i2b2 datasets.
        The resulting corpus (emrQA) has 1 million questions-logical form and 400,000+ question-answer evidence pairs.
        We characterize the dataset and explore its learning potential by training baseline models for question to
        logical form and question to answer mapping.",
    }
    """

    name = "emr_qa"
    description = "A large corpus for QA on real electronic medical records."
    tags = ["question_answering", "biomedical"]

    ALL_SPLITS: str = "all"
    SPLIT_TO_INDEX: Dict[str, int] = {
        "medication": 0,
        "relations": 1,
        "risk-dataset": 2,
        "smoking": 3,
        "obesity": 4,
    }

    def __init__(self, subset: str):
        assert subset in EmrQAScenario.SPLIT_TO_INDEX or subset == EmrQAScenario.ALL_SPLITS, f"Invalid subset: {subset}"
        self.subset: str = subset

    def get_instances(self) -> List[Instance]:
        """
        Build `Instance`s using electronic medical records and questions about them.
        """
        data_path: str = os.path.join("restricted", self.name, "dataset", "data.json")

        with open(data_path) as f:
            all_subsets = json.load(f)["data"]

        notes: List[Dict] = []
        if self.subset == EmrQAScenario.ALL_SPLITS:
            for subset in all_subsets:
                notes.extend(subset["paragraphs"])
        else:
            subset = all_subsets[EmrQAScenario.SPLIT_TO_INDEX[self.subset]]
            assert subset["title"] == self.subset
            notes = subset["paragraphs"]

        random.seed(0)
        random.shuffle(notes)
        # The authors used a 80:20 train-test split
        num_train_instances = int(len(notes) * 0.8)

        instances: List[Instance] = []
        for i, note in enumerate(notes):
            split: str = TRAIN_SPLIT if i < num_train_instances else TEST_SPLIT

            # TODO: double check that it's okay to just get first ones -Tony
            context: str = note["context"] if type(note["context"]) == str else " ".join(note["context"])
            question_answer = note["qas"][0]
            question: str = question_answer["question"][0] + "?"

            input: str = f"{context}\n{question}"
            # TODO: this can be an empty list? - no answers?
            # TODO: Answer sometimes can be an empty string?
            answer: str = question_answer["answers"][0]["text"]

            instances.append(
                Instance(input=input, references=[Reference(output=answer, tags=[CORRECT_TAG])], split=split)
            )

        return instances
