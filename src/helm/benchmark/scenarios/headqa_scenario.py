import os
from typing import List, Optional

from datasets import DatasetDict, load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_directory_exists


class HeadQAScenario(Scenario):
    """
    From "HEAD-QA: A Healthcare Dataset for Complex Reasoning" (Vilares et al.), HEAD-QA is a multi-choice
    question-answering dataset designed to evaluate reasoning on challenging healthcare-related questions.
    The questions are sourced from Spanish healthcare exams for specialized positions, covering various topics
    such as Medicine, Nursing, Psychology, Chemistry, Pharmacology, and Biology.

    Example from the dataset:

    Question:
    The excitatory postsynaptic potentials:

    A) They are all or nothing.
    B) They are hyperpolarizing.
    C) They can be added.
    D) They spread long distances.

    Answer:
    The answer is C. Explanation: None provided in this dataset.

    @InProceedings{HEAD-QA,
    author = {David Vilares and Manuel Vilares and Carlos Gómez-Rodríguez},
    title = {HEAD-QA: A Healthcare Dataset for Complex Reasoning},
    year = {2019},
    abstract = {We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex
    reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system,
    and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and
    cross-lingual (to English) experiments with information retrieval and neural techniques. We show that:
    (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance,
    demonstrating its usefulness as a benchmark for future work.}}


    Task:
    Given a question and its multiple-choice answers, models must identify the correct answer, corresponding to the
    `ra` field in the dataset. The dataset spans six healthcare domains and is challenging even for experts.
    """

    HUGGING_FACE_DATASET_PATH: str = "dvilares/head_qa"
    SKIP_VQA: bool = True
    SKIP_TEXTQA: bool = False

    name = "head_qa"
    description = "A collection of biomedical multiple-choice questions for testing medical knowledge."
    tags = ["question_answering", "biomedical", "medicine"]

    def __init__(self, language: str = "en", category: Optional[str] = None):
        """Initialize the HEAD-QA scenario.

        Args:
            language (str, optional): Language of the dataset. Defaults to "en".
            category (str, optional): Category of the dataset. If None, all categories are used.
        """
        super().__init__()
        self.language: str = language
        self.category: Optional[str] = category
        assert (
            self.SKIP_VQA or self.SKIP_TEXTQA
        ), "Failed to initialize HeadQAScenario, one of `SKIP_VQA` or `SKIP_TEXTQA` must be True."

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(self.HUGGING_FACE_DATASET_PATH, self.language)

        # XXX: Should we consider validation as test too?
        # splits = {TRAIN_SPLIT: ["train", "validation"], TEST_SPLIT: ["test"]}
        # Limit to zero shot setting
        splits = {TEST_SPLIT: ["test"]}
        instances: List[Instance] = []
        for (
            helm_split_name,
            dataset_splits_name,
        ) in splits.items():  # Iterate over the splits
            for dataset_split_name in dataset_splits_name:
                split_data = dataset[dataset_split_name]

                for example in split_data:
                    # Whether to process Visual Question Answering (VQA) examples
                    if self.SKIP_VQA and example["image"] is not None:
                        continue

                    # Whether to process Text Question Answering (TextQA) examples
                    if self.SKIP_TEXTQA and example["image"] is None:
                        continue

                    # If specified, filter by category
                    if self.category is not None:
                        if example["category"] != self.category:
                            continue

                    question = example["qtext"]

                    # Format the final answer with explanation
                    instances.append(
                        Instance(
                            input=Input(text=question),
                            references=[
                                Reference(
                                    Output(text=option["atext"]),
                                    tags=[CORRECT_TAG] if option["aid"] == example["ra"] else [],
                                )
                                for option in example["answers"]
                            ],
                            split=helm_split_name,
                            extra_data={
                                "id": example["qid"],
                                "name": example["name"],
                                "category": example["category"],
                                "year": example["year"],
                            },
                        )
                    )

        return instances
