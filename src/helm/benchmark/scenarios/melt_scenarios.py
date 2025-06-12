import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset
from helm.common.general import ensure_directory_exists
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
        subset: Optional[str] = None,
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
        subset: Optional[str] = None,
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
            revision="f8edc7f8e2873e8b271391d4489c1eedc5456f40",
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
        subset: Optional[str] = None,
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
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
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
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.text_key = "Sentence"
        self.label_key = "Emotion"

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
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.text_key = "text"
        self.label_key = "label"

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
            splits={
                TRAIN_SPLIT: "train",
                TEST_SPLIT: "test",
            },
        )
        self.text_key = "Data"
        self.label_key = "Class"

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
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.text_key = "text"
        self.label_key = "label"

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """

        return sample[self.text_key], [sample[self.label_key].lower()]


class MELTToxicityDetectionViHSDScenario(MELTTextClassificationScenario):
    """
    Scenario for the UIT-ViHSD dataset.
    """

    name = "melt_toxicity_detection_vihsd"
    description = "UIT-ViHSD dataset for toxicity detection."
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/UIT-ViHSD",
            revision="16c4f67cf509d4f9f36ca5b63c5503c7c8830557",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.label_mapping = {
            0: "clean",
            1: "offensive",
            2: "hate",
        }
        self.text_key = "free_text"
        self.label_key = "label_id"

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """
        label = sample[self.label_key]
        return sample[self.text_key], [self.label_mapping[label]]


class MELTToxicityDetectionViCTSDScenario(MELTTextClassificationScenario):
    """
    Scenario for the UIT-ViCTSD dataset.
    """

    name = "melt_toxicity_detection_victsd"
    description = "UIT-ViCTSD dataset for toxicity detection."
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__(
            dataset_name="tarudesu/ViCTSD",
            revision="65a073f2c48401410b264213229a6c52417f367a",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.label_mapping = {
            0: "clean",
            1: "toxic",
        }
        self.text_key = "Comment"
        self.label_key = "Toxicity"

    def process_example(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """
        label = sample[self.label_key]
        return sample[self.text_key], [self.label_mapping[label]]
