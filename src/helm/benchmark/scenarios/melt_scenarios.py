import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Optional

import random
from datasets import load_dataset, Dataset
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import htrack_block
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
from helm.benchmark.scenarios.math_scenario import get_answer


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


class MELTMATHScenario(Scenario):
    """
    The MATH dataset from the paper
    "Measuring Mathematical Problem Solving With the MATH Dataset"
    by Hendrycks et al. (2021):
    https://arxiv.org/pdf/2103.03874.pdf

    Example input, using official examples:

    ```
    Given a mathematics problem, determine the answer. Simplify your answer as much as possible.
    ###
    Problem: What is $\left(\frac{7}{8}\right)^3 \cdot \left(\frac{7}{8}\right)^{-3}$?
    Answer: $1$
    ###
    Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
    Answer: $15$
    ###
    Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
    Answer: $\sqrt{59}$
    ###
    Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
    Answer: $\frac{1}{32}$
    ###
    Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
    Answer: $181$
    ###
    Problem: Calculate $6 \cdot 8\frac{1}{3}
    Answer: $50$
    ###
    Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
    Answer: $2$
    ###
    Problem: How many zeros are at the end of the product 25 $\times$ 240?
    Answer: $3$
    ###
    Problem: What is $\dbinom{n}{n}$ for any positive integer $n$?
    Answer: $
    ```

    Example expected output

    ```
    1$
    ```
    """  # noqa

    name = "MATH"
    description = "Mathematical Problem Solving in Vietnamese"
    tags = ["knowledge", "reasoning"]

    subjects_mapping = {
        "number_theory": "Number Theory",
        "intermediate_algebra": "Intermediate Algebra",
        "algebra": "Algebra",
        "prealgebra": "Prealgebra",
        "geometry": "Geometry",
        "counting_and_probability": "Counting & Probability",
        "precalculus": "Precalculus",
    }
    levels = ["1", "2", "3", "4", "5"]

    def __init__(
        self, subject: str, level: str, use_official_examples: bool = False, use_chain_of_thought: bool = False
    ):
        super().__init__()
        self.subject_name: str = MELTMATHScenario.subjects_mapping[subject]
        self.subject: str = subject
        self.level: str = f"Level {level}"
        self.use_official_examples: bool = use_official_examples
        self.use_chain_of_thought: bool = use_chain_of_thought
        if use_chain_of_thought:
            assert not use_official_examples, "Cannot use official examples when use_chain_of_thought is True."

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = {}
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = load_dataset(
            "ura-hcmut/Vietnamese-MATH",
            self.subject,
            trust_remote_code=True,
            cache_dir=cache_dir,
            revision="4ee16aadb78aef3b1337e0a7267da565862673ae",
        )

        instances = []
        for split, split_name in zip([TRAIN_SPLIT, TEST_SPLIT], ["train", "test"]):
            if split == TRAIN_SPLIT and self.use_official_examples:
                train_instances = [
                    ("Kết quả của $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$ là gì?", "1"),
                    (
                        "Có bao nhiêu cách chọn 4 quyển sách từ một kệ sách có 6 quyển,"
                        + " nếu thứ tự các cuốn sách được chọn không quan trọng?",
                        "15",
                    ),
                    ("Tìm khoảng cách giữa các điểm $(2,1,-4)$ và $(5,8,-3).$", "\sqrt{59}"),
                    (
                        "Các mặt của khối xúc xắc bát diện được dán nhãn bằng các số từ $1$ đến $8$."
                        + " Xác suất tung một cặp xúc xắc bát diện để được tổng số bằng $15$ là bao nhiêu?"
                        + " Biểu diễn kết quả dưới dạng phân số tối giản.",
                        "\\frac{1}{32}",
                    ),
                    (
                        "Ba số hạng đầu tiên của một dãy số cộng lần lượt là 1, 10 và 19."
                        + " Giá trị của số hạng thứ 21 là?",
                        "181",
                    ),
                    ("Tính $6 \\cdot 8\\frac{1}{3}", "50"),
                    (
                        "Khi chia số nhị phân $100101110010_2$ cho 4,"
                        + " phần dư của phép chia là bao nhiêu (biểu diễn kết quả với cơ số 10)?",
                        "2",
                    ),
                    ("Có bao nhiêu số 0 ở cuối kết quả của tích 25 $\\times$ 240?", "3"),
                ]
                dataset[TRAIN_SPLIT] = [
                    {"problem_vi": problem, "answer_vi": answer} for problem, answer in train_instances
                ]

            else:
                examples = dataset[split].filter(lambda example: example["level"] == self.level)
                list_answers = []

                for example in examples:
                    # Sanity check that we filtered correctly
                    assert (
                        example["type"] == self.subject_name and example["level"] == self.level
                    ), f"Wrong example was included after filtering: {example}"

                    if self.use_chain_of_thought:
                        answer = example["solution_vi"]
                    else:
                        maybe_answer = get_answer(example["solution_vi"])
                        if maybe_answer is None:
                            maybe_answer = "Không có đáp án"
                        answer = maybe_answer
                    list_answers.append(answer)

                # Add column answer_vi to examples
                dataset[split] = examples.add_column("answer_vi", list_answers)

            for example in dataset[split]:
                instance = Instance(
                    input=Input(text=example["problem_vi"]),
                    references=[Reference(Output(text=example["answer_vi"]), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances


class MELTTextClassificationScenario(Scenario):
    name = "melt_text_classification"
    description = "Text Classification scenario."
    tags = ["text_classification"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        subset: Optional[str] = "",
        text_key: str = "text",
        label_key: str = "label",
        splits: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            subset: The subset of the dataset to use. Defaults to "".
            text_key: The key to use for the text in the dataset. Defaults to "text".
            label_key: The key to use for the label in the dataset. Defaults to "label".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.text_key = text_key
        self.label_key = label_key
        self.splits = splits

    @abstractmethod
    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """
        pass

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
                    input=Input(text=prompt),
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


class MELTTextClassificationVSMECScenario(MELTTextClassificationScenario):
    """
    Scenario for the UIT-VSMEC dataset.
    The UIT-VSMEC dataset is a Vietnamese emotion-labeled corpus consisting of
    6,927 human-annotated sentences collected from social media, categorized
    into six emotions: sadness, enjoyment, anger, disgust, fear, and surprise.
    """

    name = "melt_text_classification_vsmec"
    description = "UIT-VSMEC dataset for emotion classification."
    tags = ["text_classification"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/UIT-VSMEC",
            revision="ab642b189eff31fdb781cca7c4c34dee3ee0f1de",
            text_key="Sentence",
            label_key="Emotion",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """

        return sample[self.text_key], [sample[self.label_key].lower()]


class MELTTextClassificationPhoATISScenario(MELTTextClassificationScenario):
    """
    Scenario for the PhoATIS dataset.
    The PhoATIS dataset is a Vietnamese benchmark for intent detection and slot filling,
    consisting of 5,871 fluent utterances collected from task-oriented dialogue systems.
    It was later extended with manual disfluency annotations to create a disfluent variant,
    enabling research on the impact of disfluencies in spoken language understanding for Vietnamese.
    """

    name = "melt_text_classification_phoatis"
    description = "PhoATIS dataset for intent detection of flight booking."
    tags = ["text_classification"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/PhoATIS",
            revision="bd026c9b276d7fb083d19ec3d6870fca90e1834f",
            text_key="text",
            label_key="label",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """

        return sample[self.text_key], sample[self.label_key].lower().split("#")


class MELTTSentimentAnalysisVLSPScenario(MELTTextClassificationScenario):
    """
    Scenario for the VLSP 2016 sentiment analysis dataset.
    The VLSP2016 dataset is a Vietnamese sentiment analysis corpus consisting of
    short user-generated reviews from social media, each labeled with an overall
    sentiment of positive, negative, or neutral. It was developed to support polarity
    classification and benchmark Vietnamese sentiment analysis systems through the
    VLSP 2016 evaluation campaign.
    """

    name = "melt_sentiment_analysis_vlsp"
    description = "VLSP 2016 contains public comments from social media, used for sentiment analysis."
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/vlsp2016",
            revision="9531ec0ccabcafb7d51020fe69d8f9faebb91953",
            text_key="Data",
            label_key="Class",
            splits={
                TRAIN_SPLIT: "train",
                TEST_SPLIT: "test",
            },
        )

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """

        return sample[self.text_key], [sample[self.label_key].lower()]


class MELTTSentimentAnalysisVSFCScenario(MELTTextClassificationScenario):
    """
    Scenario for the UIT-VSFC dataset.
    The UIT-VSFC dataset is a Vietnamese corpus of over 16,000 student feedback sentences,
    annotated for both sentiment-based (positive, negative, neutral) and topic-based classifications.
    It supports interdisciplinary research at the intersection of sentiment analysis and education,
    with high inter-annotator agreement and strong baseline performance using a Maximum Entropy classifier.
    """

    name = "melt_sentiment_analysis_vsfc"
    description = "UIT-VSFC dataset for analyzing sentiment of student feedback."
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/UIT-VSFC",
            revision="c572aed01a811a1dbc68e9aed9f9e684980a10a2",
            text_key="text",
            label_key="label",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """

        return sample[self.text_key], [sample[self.label_key].lower()]


class MELTTranslationScenario(Scenario):
    """ """

    name = "melt_translation"
    description = "Machine Translation scenario."
    tags = ["machine_translation"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        source_language: str,
        target_language: str,
        subset: Optional[str] = "",
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            source_language: The source language to use.
            target_language: The target language to use.
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.MAX_TRAIN_INSTANCES = 20_000
        valid_languages = set(["vi", "en"])
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.splits = splits
        self.source_language = source_language
        self.target_language = target_language
        if self.source_language not in valid_languages or self.target_language not in valid_languages:
            raise ValueError("Supported languages: vi, en.")
        if self.source_language == self.target_language:
            raise ValueError("The source language and the target language should be different.")
        if self.source_language != "en" and self.target_language != "en":
            raise ValueError("One of the languages should be English.")

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        with htrack_block("Loading the HuggingFace dataset. The first time could take several minutes."):
            hf_dataset: Any = load_dataset(
                self.dataset_name,
                self.subset,
                revision=self.revision,
                trust_remote_code=True,
            )

        instances: List[Instance] = []

        for dataset_split_name, helm_split_name in splits.items():
            if helm_split_name == TRAIN_SPLIT:
                hf_dataset[dataset_split_name] = hf_dataset[dataset_split_name].shuffle(seed=42)[
                    : self.MAX_TRAIN_INSTANCES
                ]
                hf_dataset[dataset_split_name] = Dataset.from_dict(hf_dataset[dataset_split_name])

            for example in hf_dataset[dataset_split_name]:
                source_sentence = example[self.source_language]
                target_sentence = example[self.target_language]
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
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


class MELTTranslationOPUS100Scenario(MELTTranslationScenario):
    """
    Scenario for the OPUS100 dataset.
    """

    name = "melt_translation_opus100"
    description = "OPUS100 dataset for machine translation."
    tags = ["machine_translation"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="vietgpt/opus100_envi",
            revision="45df06fb0b31edc882d7c8d34389261f995e5208",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )


class MELTTranslationPhoMTScenario(MELTTranslationScenario):
    """
    Scenario for the PhoMT dataset.
    """

    name = "melt_translation_phomt"
    description = "PhoMT dataset for machine translation."
    tags = ["machine_translation"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="ura-hcmut/PhoMT",
            revision="74386685db01dc038860ff0a90d9f5fbde284bf7",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )


class MELTLMMaskFillingScenario(Scenario):
    """
    Scenario for the MELT Masked Language Modeling dataset.
    """

    name = "melt_lm_mask_filling"
    description = "Masked Language Modeling scenario."
    tags = ["language_modeling", "mask_filling"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        masked_ratio: float = 0.1,
        text_key: str = "text",
        subset: Optional[str] = "",
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            masked_ratio: The ratio of tokens to mask in the input text. Defaults to 0.1.
            text_key: The key to use for the text in the dataset. Defaults to "text".
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.masked_ratio = masked_ratio
        self.text_key = text_key
        self.revision = revision
        self.splits = splits

    def __mask_text__(self, text: str) -> str:
        """
        Mask a portion of the input text.
        Args:
            text (str): The input text to mask.
        Returns:
            str: The masked text.
        """
        tokens = text.split(" ")
        num_tokens_to_mask = int(len(tokens) * self.masked_ratio)
        indices_to_mask = random.sample(range(len(tokens)), num_tokens_to_mask)
        for index in indices_to_mask:
            tokens[index] = "[MASK]"
        return " ".join(tokens)

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset: Any = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )

        for dataset_split_name, helm_split_name in splits.items():
            for sample in dataset[dataset_split_name]:
                target_sentence = sample[self.text_key]
                source_sentence = self.__mask_text__(target_sentence)
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
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


class MELTLMMaskFillingMLQAScenario(MELTLMMaskFillingScenario):
    """
    Scenario for the MLQA dataset.
    This dataset is a multilingual question answering dataset.
    It contains questions in multiple languages and their corresponding
    answers in the same language. In this scenario, we are using the
    context of questions in the Vietnamese subset of the MLQA dataset.
    """

    name = "melt_lm_mask_filling_mlqa"
    description = "MLQA dataset for masked language modeling."
    tags = ["language_modeling", "mask_filling"]

    def __init__(self):
        super().__init__(
            dataset_name="facebook/mlqa",
            revision="397ed406c1a7902140303e7faf60fff35b58d285",
            subset="mlqa.vi.vi",
            text_key="context",
            splits={
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )


class MELTLMSpellingCorrectionScenario(Scenario):
    """
    Scenario for the MELT spelling correction dataset.
    """

    name = "melt_lm_spelling_correction"
    description = "Spelling Correction scenario."
    tags = ["language_modeling", "spelling_correction"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        source_key: str = "text",
        target_key: str = "corrected_text",
        subset: Optional[str] = "",
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            source_key: The key to use for the source text in the dataset. Defaults to "text".
            target_key: The key to use for the target text in the dataset. Defaults to "corrected_text".
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.source_key = source_key
        self.target_key = target_key
        self.revision = revision
        self.splits = splits

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset: Any = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        if len(splits) == 1:
            all_keys = list(splits.keys())
            dataset = dataset[all_keys[0]].train_test_split(test_size=0.33)
            splits = {
                "train": TRAIN_SPLIT,
                "test": TEST_SPLIT,
            }

        for dataset_split_name, helm_split_name in splits.items():
            for sample in dataset[dataset_split_name]:
                source_sentence = sample[self.source_key]
                target_sentence = sample[self.target_key]
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
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


class MELTLMSpellingCorrectionVSECScenario(MELTLMSpellingCorrectionScenario):
    """
    Scenario for the VSEC dataset.
    The VSEC dataset is a Vietnamese spelling correction dataset.
    It contains 9,341 pairs of sentences where the first sentence is a misspelled
    version of the second sentence, which is the correct version.
    The mistakes are common spelling errors made by Vietnamese speakers and typists.
    """

    name = "melt_lm_spelling_correction_vsec"
    description = "VSEC dataset for spelling correction."
    tags = ["language_modeling", "spelling_correction"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/VSEC",
            revision="a6732e131605b5ec24ecc1745c6061c5ae86814e",
            source_key="text",
            target_key="correct",
            splits={
                TEST_SPLIT: "test",
            },
        )
