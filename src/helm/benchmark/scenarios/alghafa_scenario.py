import os
from typing import Dict, List

import datasets

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    TRAIN_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class AlGhafaScenario(Scenario):
    """AlGhafa Evaluation Benchmark for Arabic Language Models

    EXPERIMENTAL: This scenario may have future reverse incompatible changes.

    Multiple-choice evaluation benchmark for zero- and few-shot evaluation of Arabic LLMs,
    consisting of

    - https://huggingface.co/datasets/OALL/AlGhafa-Arabic-LLM-Benchmark-Native/
    - https://aclanthology.org/2023.arabicnlp-1.21/

    Citation:

    ```
    @inproceedings{almazrouei-etal-2023-alghafa,
        title = "{A}l{G}hafa Evaluation Benchmark for {A}rabic Language Models",
        author = "Almazrouei, Ebtesam  and
        Cojocaru, Ruxandra  and
        Baldo, Michele  and
        Malartic, Quentin  and
        Alobeidli, Hamza  and
        Mazzotta, Daniele  and
        Penedo, Guilherme  and
        Campesan, Giulia  and
        Farooq, Mugariya  and
        Alhammadi, Maitha  and
        Launay, Julien  and
        Noune, Badreddine",
        editor = "Sawaf, Hassan  and
        El-Beltagy, Samhaa  and
        Zaghouani, Wajdi  and
        Magdy, Walid  and
        Abdelali, Ahmed  and
        Tomeh, Nadi  and
        Abu Farha, Ibrahim  and
        Habash, Nizar  and
        Khalifa, Salam  and
        Keleg, Amr  and
        Haddad, Hatem  and
        Zitouni, Imed  and
        Mrini, Khalil  and
        Almatham, Rawan",
        booktitle = "Proceedings of ArabicNLP 2023",
        month = dec,
        year = "2023",
        address = "Singapore (Hybrid)",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.arabicnlp-1.21/",
        doi = "10.18653/v1/2023.arabicnlp-1.21",
        pages = "244--275",
        abstract = "Recent advances in the space of Arabic large language models have opened up a wealth of potential practical applications. From optimal training strategies, large scale data acquisition and continuously increasing NLP resources, the Arabic LLM landscape has improved in a very short span of time, despite being plagued by training data scarcity and limited evaluation resources compared to English. In line with contributing towards this ever-growing field, we introduce AlGhafa, a new multiple-choice evaluation benchmark for Arabic LLMs. For showcasing purposes, we train a new suite of models, including a 14 billion parameter model, the largest monolingual Arabic decoder-only model to date. We use a collection of publicly available datasets, as well as a newly introduced HandMade dataset consisting of 8 billion tokens. Finally, we explore the quantitative and qualitative toxicity of several Arabic models, comparing our models to existing public Arabic LLMs."
    }
    ```
    """  # noqa: E501

    name = "alghafa"
    description = "AlGhafa"
    tags = ["multiple choice"]

    HF_SPLIT_TO_HELM_SPLIT = {"validation": TRAIN_SPLIT, "test": TEST_SPLIT}
    REFERENCE_PREFIX = "sol"

    def __init__(self, subset: str):
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset_splits: Dict[str, datasets.Dataset] = datasets.load_dataset(
            "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            self.subset,
            revision="a31ebd34ca311d7e0cfc6ad7f458b3435af280f5",
            cache_dir=cache_dir,
        )

        # Read all instances
        instances: List[Instance] = []
        for split_name, dataset in dataset_splits.items():
            assert isinstance(dataset, datasets.Dataset)
            option_indexes = [
                int(s.removeprefix(self.REFERENCE_PREFIX))
                for s in dataset[0].keys()
                if s.startswith(self.REFERENCE_PREFIX)
            ]
            for row_index, row in enumerate(dataset):
                input = Input(text=row["query"])
                references: List[Reference] = []
                # Need to add 1 because label is zero-indexed and has a value from 0 to (N - 1),
                # but column names are 1 indexed and have values from "sol1" to "solN"
                correct_option_index = int(row["label"]) + 1
                for option_index in option_indexes:
                    column_name = f"{self.REFERENCE_PREFIX}{option_index}"
                    references.append(
                        Reference(
                            output=Output(text=row[column_name]),
                            tags=[CORRECT_TAG] if option_index == correct_option_index else [],
                        )
                    )
                instance = Instance(
                    id=f"id{row_index}_{split_name}",
                    input=input,
                    references=references,
                    split=self.HF_SPLIT_TO_HELM_SPLIT[split_name],
                )
                instances.append(instance)

        return instances
