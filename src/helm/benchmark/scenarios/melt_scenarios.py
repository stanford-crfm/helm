from typing import Dict, List, Tuple, Optional

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Input,
    Output,
)


class MELTQAScenario(Scenario):
    name = "melt_question_answering"
    description = "Question answering scenario."
    tags = ["question_answering"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        subset: Optional[str] = "",
        passage_prefix: str = "Passage: ",
        question_prefix: str = "Question: ",
        splits: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            subset: The subset of the dataset to use. Defaults to "".
            passage_prefix: The prefix to use for the context passage. Defaults to "Passage: ".
            question_prefix: The prefix to use for the question. Defaults to "Question: ".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.passage_prefix = passage_prefix
        self.question_prefix = question_prefix
        self.splits = splits

    def process_example(self, sample: dict) -> Tuple[Input, List[str]]:
        """
        Given an sample from the dataset, create the prompt and the list of
        correct references.
        Each sample is a dictionary with the following keys:
        - context: The passage to be used for the question.
        - question: A questions.
        - answers: A list of answers with dictionary format {'answer_start': [], 'text': []}
        """
        passage = sample["context"]
        question = sample["question"]
        prompt = PassageQuestionInput(
            passage=passage,
            passage_prefix=self.passage_prefix,
            question=question,
            question_prefix=self.question_prefix,
            separator="\n\n",
        )

        answers: List[str] = []
        for answer_text in sample["answers"]["text"]:
            answers.append(answer_text)

        return prompt, answers

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        for dataset_split_name, helm_split_name in splits.items():
            if dataset_split_name not in dataset:
                raise ValueError(f"Could not find split {dataset_split_name} in dataset {self.dataset_name}")

            for sample in dataset[dataset_split_name]:
                prompt, answers = self.process_example(sample)
                instance = Instance(
                    input=prompt,
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                    split=helm_split_name,
                )
                instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.splits is None:
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            splits = {}
            if "train" in self.splits:
                splits[self.splits[TRAIN_SPLIT]] = TRAIN_SPLIT
            if "validation" in self.splits:
                splits[self.splits[VALID_SPLIT]] = VALID_SPLIT
            if "test" in self.splits:
                splits[self.splits[TEST_SPLIT]] = TEST_SPLIT

        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTQAMLQAScenario(MELTQAScenario):
    """
    Scenario for MLQA dataset.
    This dataset is a multilingual question answering dataset.
    It contains questions in multiple languages and their corresponding
    answers in the same language.
    In this scenario, we are using the Vietnamese subset of the MLQA dataset.
    """

    name = "melt_question_answering_mlqa"
    description = "MLQA is an open-ended question answering dataset in multiple languages."
    tags = ["question_answering"]

    def __init__(self):
        super().__init__(
            dataset_name="facebook/mlqa",
            revision="397ed406c1a7902140303e7faf60fff35b58d285",
            subset="mlqa.vi.vi",
            passage_prefix="Ngữ cảnh: ",
            question_prefix="Câu hỏi: ",
            splits={
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )


class MELTQAXQuADScenario(MELTQAScenario):
    """
    Scenario for XQuAD dataset.
    XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset
    for evaluating cross-lingual question answering performance.
    """

    name = "melt_question_answering_xquad"
    description = "XQuAD is a cross-lingual question answering dataset."
    tags = ["question_answering"]

    def __init__(self):
        super().__init__(
            dataset_name="juletxara/xquad_xtreme",
            revision="87646a09233481f6884b6ffcc6795af9ca0b85d7",
            subset="vi",
            passage_prefix="Ngữ cảnh: ",
            question_prefix="Câu hỏi: ",
            splits={
                VALID_SPLIT: "translate_train",
                TEST_SPLIT: "test",
            },
        )


class MELTSummarizationScenario(Scenario):
    """
    Scenario for single document text summarization.
    """

    name = "melt_summarization"
    description = "Scenario for summarization tasks"
    tags = ["summarization"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        subset: Optional[str] = "",
        train_min_length: Optional[int] = None,
        train_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
        article_key: str = "source",
        summary_key: str = "target",
        splits: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes summarization scenario.
        Args:
            dataset_name: String identifier for dataset. Currently
                          supported options ["vietnews", "wikilingua"].
            revision: String identifier for dataset version.
            subset: Dataset subset to use. Defaults to "".
            train_min_length: Int indicating minimum length for training
                                 documents. Training examples smaller than
                                 train_min_length will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small documents.
            train_max_length: Int indicating maximum length for training
                                 documents. Training examples larger than
                                 train_max_length will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large documents.
            doc_max_length: Int indicating the maximum length to truncate
                            documents. Documents in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Currently uses whitespace tokenization.
            article_key: String key for article text in dataset. Defaults to "source".
            summary_key: String key for summary text in dataset. Defaults to "target".
            splits: Dict containing split names and corresponding split. If
                    None, defaults to {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.revision = revision
        self.subset = subset
        self.train_min_length = train_min_length
        self.train_max_length = train_max_length
        self.doc_max_length = doc_max_length
        self.article_key = article_key
        self.summary_key = summary_key
        self.splits = splits

    def _clean_and_truncate(self, text: str, max_length: Optional[int] = None) -> str:
        return " ".join(text.split()[:max_length])

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )

        for dataset_split_name, helm_split_name in splits.items():
            if dataset_split_name not in dataset:
                raise ValueError(f"Could not find split {dataset_split_name} in dataset {self.dataset_name}")

            for sample in dataset[dataset_split_name]:
                article: str = self._clean_and_truncate(sample[self.article_key], self.doc_max_length)
                summary: str = self._clean_and_truncate(sample[self.summary_key])

                if helm_split_name == "train":
                    art_len = len(article.split())
                    if self.train_max_length and art_len > self.train_max_length:
                        continue
                    if self.train_min_length and art_len < self.train_min_length:
                        continue

                instances.append(
                    Instance(
                        input=Input(text=article),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=helm_split_name,
                    )
                )

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.splits is None:
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            splits = {}
            if "train" in self.splits:
                splits[self.splits[TRAIN_SPLIT]] = TRAIN_SPLIT
            if "validation" in self.splits:
                splits[self.splits[VALID_SPLIT]] = VALID_SPLIT
            if "test" in self.splits:
                splits[self.splits[TEST_SPLIT]] = TEST_SPLIT

        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTSummarizationVietnewsScenario(MELTSummarizationScenario):
    """
    Scenario for summarization on Vietnews dataset.
    Vietnews includes a collection of news articles in Vietnamese from
    online news such as Tuoi Tre, VnExpress, and Nguoi Dua Tin between 2016 and 2019.
    The topic of the articles is about the world, news, law, and business.
    The dataset also contains the corresponding summary for each article.
    """

    name = "melt_summarization_vietnews"
    description = (
        "Vietnews is a Vietnamese news summarization dataset collected from online news articles between 2016 and 2019."
    )
    tags = ["summarization"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="Yuhthe/vietnews",
            revision="c391150e7541839d0f07d9ea89fe00005618a8f7",
            article_key="article",
            summary_key="abstract",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )


class MELTSummarizationWikilinguaScenario(MELTSummarizationScenario):
    """
    Scenario for summarization on Wikilingua dataset.
    Wikilingua is a multilingual summarization dataset.
    In this scenario, we are using the Vietnamese subset of the Wikilingua dataset.
    """

    name = "melt_summarization_wikilingua"
    description = "Wikilingua is a multilingual summarization dataset."
    tags = ["summarization"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="GEM/wiki_lingua",
            revision="af5d0f00b59a6933165c97b384f50d8b563c314d",
            article_key="source",
            summary_key="target",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )
