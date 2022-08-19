import os
import random
import re
from typing import List

import cleantext

from common.general import ensure_directory_exists, ensure_file_downloaded, hlog
from .scenario import Scenario, Instance, Reference, ALL_SPLITS, CORRECT_TAG, VALID_SPLIT


class MeQSumScenario(Scenario):
    """
    From "On the Summarization of Consumer Health Questions" (Abacha et al.), MeQSum is a corpus of 1,000 summarized
    consumer health questions.

    The following is an example from the dataset:

    @Inproceedings{MeQSum,
    author = {Asma {Ben Abacha} and Dina Demner-Fushman},
    title = {On the Summarization of Consumer Health Questions},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019,
    Florence, Italy, July 28th - August 2},
    year = {2019},
    abstract = {Question understanding is one of the main challenges in question answering. In real world applications,
    users often submit natural language questions that are longer than needed and include peripheral information that
    increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this
    paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of
    1,000 summarized consumer health questions. We explore data augmentation methods and evaluate state-of-the-art
    neural abstractive models on this new task. In particular, we show that semantic augmentation from question datasets
    improves the overall performance, and that pointer-generator networks outperform sequence-to-sequence attentional
    models on this task, with a ROUGE-1 score of 44.16%. We also present a detailed error analysis and discuss
    directions for improvement that are specific to question summarization.}}
    """

    name = "me_q_sum"
    description = "Medical dialogue dataset "
    tags = ["summarization", "biomedical"]

    # TODO: use our clean version -Tony
    DOWNLOAD_URL: str = (
        "https://raw.githubusercontent.com/UCSD-AI4H/COVID-Dialogue/master/COVID-Dialogue-Dataset-English.txt"
    )

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        """
        Adapted from the official download script in the GitHub repository:
        https://github.com/UCSD-AI4H/COVID-Dialogue.
        """
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        dataset_path: str = os.path.join(data_path, "COVID-Dialogue-Dataset-English.txt")
        ensure_file_downloaded(source_url=MeQSumScenario.DOWNLOAD_URL, target_path=dataset_path)

        dialogues = []
        for text in open(dataset_path).read().split("id="):
            text = text[text.find("Description") :].strip()
            description: str = text[len("Description\n") : text.find("\nDialogue")]
            if description:
                description = cleantext.clean(description, extra_spaces=True, lowercase=True)

            text = text[text.find("\nPatient:") :]

            utterances, last_person, valid = [], "None", True
            for x in re.finditer("Doctor:|Patient:", text):
                if x.group() == last_person:
                    valid = False
                    break
                else:
                    last_person = x.group()

                utterance = text[x.end() :].split("Patient:")[0].split("Doctor:")[0]
                utterances.append(cleantext.clean(utterance, extra_spaces=True, lowercase=True))

            if valid and utterances:
                dialogues.append({"description": description, "utterances": utterances})

        hlog(f"Number of dialogs: {len(dialogues)}")

        # Used the exact same seed value in the official download script
        random.seed(11111)
        random.shuffle(dialogues)

        train_size = int(0.8 * len(dialogues))
        dev_size = int(0.1 * len(dialogues))
        train_split = dialogues[:train_size]
        dev_split = dialogues[train_size : train_size + dev_size]
        test_split = dialogues[train_size + dev_size :]

        import pdb

        pdb.set_trace()
        instances: List[Instance] = []

        return instances
