import os
from typing import List

from common.general import ensure_directory_exists, ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, ALL_SPLITS, CORRECT_TAG, VALID_SPLIT


class BioLeafletsScenario(Scenario):
    """
    TODO

    @inproceedings{yermakov-etal-2021-biomedical,
    title = "Biomedical Data-to-Text Generation via Fine-Tuning Transformers",
    author = "Yermakov, Ruslan  and
      Drago, Nicholas  and
      Ziletti, Angelo",
    booktitle = "Proceedings of the 14th International Conference on Natural Language Generation",
    month = aug,
    year = "2021",
    address = "Aberdeen, Scotland, UK",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.inlg-1.40",
    pages = "364--370",
    abstract = "Data-to-text (D2T) generation in the biomedical domain is a promising - yet mostly unexplored - field
    of research. Here, we apply neural models for   D2T generation to a real-world dataset consisting of package
    leaflets of European medicines. We show that fine-tuned transformers are able to generate realistic,
    multi-sentence text from data in the biomedical domain, yet have important limitations. We also release a new
    dataset (BioLeaflets) for benchmarking D2T generation models in the biomedical domain.",
    })
    """

    name = "bioleaflets"
    description = ""
    tags = ["summarization", "biomedical"]

    SOURCE_URL_TEMPLATE: str = (
        "https://worksheets.codalab.org/rest/bundles/0x7baf4168313e490bbd1cb3203308fa1f/contents/blob/{file_name}"
    )

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        """
        Build `Instance`s using the consumer health questions and their summarized versions.
        """

        def download_and_read_lines(file_name: str) -> List[str]:
            file_path: str = os.path.join(data_path, file_name)
            ensure_file_downloaded(
                source_url=BioLeafletsScenario.SOURCE_URL_TEMPLATE.format(file_name=file_name),
                target_path=file_path,
                unpack=False,
            )

            with open(file_path) as f:
                return f.read().splitlines()

        data_path: str = os.path.join(self.output_path, "data")
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
                    Instance(input=question, references=[Reference(output=summary, tags=[CORRECT_TAG])], split=split)
                )

        return instances
