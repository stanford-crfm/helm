import os
from typing import List

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
from helm.common.hierarchical_logger import hwarn


class ArabicEXAMSScenario(Scenario):
    """The Arabic subset of the EXAMS High School Examinations Dataset for Multilingual Question Answering

    We use the Open Arabic LLM Leaderboard (OALL) version mirror of the Arabic subset of EXAMS, which is in-turn based
    on the AceGPT version.

    See: https://www.tii.ae/news/introducing-open-arabic-llm-leaderboard-empowering-arabic-language-modeling-community

    References:

    ```
    @misc{huang2024acegptlocalizinglargelanguage,
        title={AceGPT, Localizing Large Language Models in Arabic},
        author={Huang Huang and Fei Yu and Jianqing Zhu and Xuening Sun and Hao Cheng and Dingjie Song and Zhihong Chen and Abdulmohsen Alharthi and Bang An and Juncai He and Ziche Liu and Zhiyi Zhang and Junying Chen and Jianquan Li and Benyou Wang and Lian Zhang and Ruoyu Sun and Xiang Wan and Haizhou Li and Jinchao Xu},
        year={2024},
        eprint={2309.12053},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2309.12053},
    }```

    ```
    @inproceedings{hardalov-etal-2020-exams,
        title = "{EXAMS}: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering",
        author = "Hardalov, Momchil  and
        Mihaylov, Todor  and
        Zlatkova, Dimitrina  and
        Dinkov, Yoan  and
        Koychev, Ivan  and
        Nakov, Preslav",
        editor = "Webber, Bonnie  and
        Cohn, Trevor  and
        He, Yulan  and
        Liu, Yang",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.emnlp-main.438/",
        doi = "10.18653/v1/2020.emnlp-main.438",
        pages = "5427--5444",
        abstract = "We propose EXAMS {--} a new benchmark dataset for cross-lingual and multilingual question answering for high school examinations. We collected more than 24,000 high-quality high school exam questions in 16 languages, covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.EXAMS offers unique fine-grained evaluation framework across multiple languages and subjects, which allows precise analysis and comparison of the proposed models. We perform various experiments with existing top-performing multilingual pre-trained models and show that EXAMS offers multiple challenges that require multilingual knowledge and reasoning in multiple domains. We hope that EXAMS will enable researchers to explore challenging reasoning and knowledge transfer methods and pre-trained models for school question answering in various languages which was not possible by now. The data, code, pre-trained models, and evaluation are available at http://github.com/mhardalov/exams-qa."
    }```
    """  # noqa: E501

    name = "arabic_exams"
    description = "EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations. "  # noqa: E501
    tags = ["knowledge", "multiple_choice"]

    CHOICES = ["A", "B", "C", "D"]
    # Remap validation split to train split
    HF_SPLIT_TO_HELM_SPLIT = {"validation": TRAIN_SPLIT, "test": TEST_SPLIT}

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject.replace("_", " ")

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset_splits = datasets.load_dataset(
            "OALL/Arabic_EXAMS",
            revision="bc7a29346dbcaa16a8cd883b1f3e681ab2b7ff2a",
            cache_dir=cache_dir,
        )

        instances: List[Instance] = []
        for split_name, dataset in dataset_splits.items():
            for row in dataset:
                subject = row["id"].split("-")[0]
                if self.subject != "all" and self.subject != subject:
                    continue
                input = Input(text=row["question"])
                references: List[Reference] = []
                if row["answer"] not in self.CHOICES:
                    hwarn(f"Invalid value in answer column in row: {row}")
                    continue
                correct_choice = row["answer"]
                for choice in self.CHOICES:
                    references.append(
                        Reference(
                            output=Output(text=row[choice]),
                            tags=[CORRECT_TAG] if choice == correct_choice else [],
                        )
                    )
                instance = Instance(
                    id=row["id"],
                    input=input,
                    references=references,
                    split=self.HF_SPLIT_TO_HELM_SPLIT[split_name],
                )
                instances.append(instance)

        return instances
