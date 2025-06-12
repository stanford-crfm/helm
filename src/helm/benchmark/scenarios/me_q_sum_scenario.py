import os
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    Input,
    Output,
)


class MeQSumScenario(Scenario):
    """
    From "On the Summarization of Consumer Health Questions" (Abacha et al.), MeQSum is a corpus of 1,000 summarized
    consumer health questions.

    The following is an example from the dataset:

    Question:
    SUBJECT: inversion of long arm chromasome7 MESSAGE: My son has been diagnosed with inversion of long arm
    chromasome 7 and down syndrome . please could you give me information on the chromasome 7 please because
    our doctors have not yet mentioned it

    Summary:
    Where can I find information on chromosome 7?

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

    SOURCE_URL_TEMPLATE: str = (
        "https://worksheets.codalab.org/rest/bundles/0xd98a53314314445b96b4d703bb2d8c8c/contents/blob/{file_name}"
    )

    name = "me_q_sum"
    description = "MeQSum is a corpus of 1,000 summarized consumer health questions."
    tags = ["summarization", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Build `Instance`s using the consumer health questions and their summarized versions.
        """

        def download_and_read_lines(file_name: str) -> List[str]:
            file_path: str = os.path.join(data_path, file_name)
            ensure_file_downloaded(
                source_url=MeQSumScenario.SOURCE_URL_TEMPLATE.format(file_name=file_name),
                target_path=file_path,
                unpack=False,
            )

            with open(file_path) as f:
                return f.read().splitlines()

        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            dataset_split: str = "val" if split == VALID_SPLIT else split

            # The files with the questions end with ".source"
            questions: List[str] = download_and_read_lines(f"{dataset_split}.source")

            # The files with the summaries end with ".target"
            summaries: List[str] = download_and_read_lines(f"{dataset_split}.target")

            for question, summary in zip(questions, summaries):
                instances.append(
                    Instance(
                        input=Input(text=question),
                        references=[Reference(output=Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
