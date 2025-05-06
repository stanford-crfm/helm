import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional, Literal

import random
import dataclasses
import numpy as np

from copy import copy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MELTLanguageLogicalStatement:
    subject: str  # e.g. either the individual or group to which this statement applies
    subject_category: str  # e.g. the group to which this fact applies
    specifier_type: Literal["một", "cái"]  # the specifier used for the subject

    def generate_specified_subject(self, upper=False, specifier_type=None) -> str:
        """Handle the specification of the subject in the statement.
        It is similar to the English "a" or "the" in the statement.

        Example:
            if (subject="con mèo", subject_category="động vật", specifier_type="cái", upper=False) -> "Cái con mèo"
            if (subject="quả táo", subject_category="thực vật", specifier_type="a", upper=True) -> "Một quả táo"
        """

        specifier_type = self.specifier_type if specifier_type is None else specifier_type
        if not (self.subject_category != "người") or (self.subject == "người"):
            return self.subject
        base_char = specifier_type[0].upper() if upper else specifier_type[0].lower()
        return f"{base_char}{specifier_type[1:]} {self.subject}"


@dataclass(frozen=True)
class MELTLanguageRule(MELTLanguageLogicalStatement):
    """Class describing how a set of attributes about an individual/group imply another attribute."""

    condition: List[str]  # a list of attributes which must apply for the rule to apply
    condition_conjunction: Literal["và", "hoặc"]  # "and" or "or", corresponding to
    consequent: str  # the attribute resulting from the application of the rule

    def __str__(self) -> str:
        """Renders the rule, i.e. corresponding to "if x (and/or y) then z"

        Rules should have the following format:
        {
            'subject': 'An',
            'subject_category': 'người',
            'specifier_type': 'cái' or 'một'
            'condition': ['đỏ', 'tốt'],
            'condition_conjunction': 'và',
            'consequent': 'cold'
        }

        and this example will output a string: "Nếu An là đỏ và tốt, thì An là lạnh."
        """

        condition = f" {self.condition_conjunction} ".join(self.condition)
        specified_subject = self.generate_specified_subject()
        specified_particular_subject = self.generate_specified_subject(specifier_type="cái")
        return f"Nếu {specified_subject} là {condition}, thì {specified_particular_subject} là {self.consequent}."


@dataclass(frozen=True)
class MELTLanguageFact(MELTLanguageLogicalStatement):
    """Class describing a statement that a subject has some attributes."""

    specific_attributes: List[str]  # more specific versions of the attributes
    generic_attributes: List[str]  # a list of attributes which apply to the subject
    use_specific_attributes: bool  # whether to use the more specific attributes (i.e. hard mode)
    upper: bool = True  # whether the statement should be uppercase

    def __str__(self) -> str:
        """Maps from a set of attributes about a subject to a string

        e.g. if (subject="con chó", attributes=["to", "đỏ"], specifier="cái") ->
        "Cái con chó thì to và đỏ."
        """

        if len(self.generic_attributes) == 0:
            return "Không có gì."
        target_attributes = self.specific_attributes if self.use_specific_attributes else self.generic_attributes
        specified_subject = self.generate_specified_subject(upper=self.upper)
        return f"{specified_subject} là {' và '.join(target_attributes)}."


def get_vocab() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """All potential subjects for the facts and rules for sythetic_reasoning_natural as well as their categories.
    Subjects is a dictionary of subject categories like "người" and "động vật" which correspond to
    a list of potential subjects.

    Attributes corresponds to an initial list of attributes which are only synonymous with themselves.
    Intially, we default to not including these attributes, but we leave this feature in for convenience.

    Attribute groups are a more general version of attributes, where a single attribute corresponds to a class of
    attributes e.g. if we know something is chilly, we know that it is cold (but not assuming the reverse).
    """

    # A list of subjects and their categories
    subjects: Dict[str, List[str]] = {
        "người": ["An", "Bình", "Cường", "Duy", "Đạt", "Phương"],
        "động vật": [
            "con chó",
            "con mèo",
            "con thỏ",
            "con chuột",
            "con hổ",
            "con sư tử",
            "con gấu",
            "con sóc",
            "con bò",
            "con gấu trúc",
            "con nhím",
            "con voi",
            "con hươu cao cổ",
            "con hà mã",
        ],
        "thực vật": ["hoa anh túc", "hoa bồ công anh", "cây", "hoa hồng", "hoa hướng dương"],
    }

    # Convert list of attributes into dictionary
    # A list of attributes and their overarching meaning (used in hard difficulty)
    attribute_groups = {
        "trẻ": ["trẻ"],
        "mềm": ["mềm"],
        "buồn": ["buồn"],
        "sợ": ["sợ"],
        "lạnh": ["lạnh", "lạnh buốt", "mát mẻ"],
        "nóng": ["nóng", "ấm"],
        "thông minh": ["thông minh", "tài giỏi", "khôn", "sáng trí"],
        "sạch": ["sạch", "ngăn nắp"],
        "nhỏ": ["nhỏ", "bé", "tí nị"],
        "to": ["to", "khổng lồ", "bự", "lớn"],
        "tốt": ["tốt", "tử tế", "tốt bụng"],
        "đẹp": ["đẹp", "xinh"],
        "đỏ": ["đỏ", "đỏ thẫm"],
        "xanh dương": ["xanh dương", "xanh lam"],
        "xanh lục": ["xanh lục", "xanh lá cây"],
        "tím": ["tím", "tím than"],
        "chán": ["chán", "đần"],
        "cũ": ["cũ", "xưa", "cổ"],
        "mạnh": ["mạnh", "mạnh mẽ", "cơ bắp"],
        "yếu": ["yếu", "yếu đuối", "mỏng manh"],
        "nhanh": ["nhanh", "mau"],
        "chậm": ["chậm", "chậm chạp"],
        "xấu": ["xấu", "xấu xa", "ác", "độc ác"],
        "hạnh phúc": ["hạnh phúc", "hân hoan", "vui mừng", "vui vẻ"],
        "tròn": ["tròn", "hình tròn", "hình cầu"],
    }
    # Remove any keys which duplicate subitems
    new_attribute_groups: Dict[str, List[str]] = copy(attribute_groups)
    for general_attribute, specific_attributes in attribute_groups.items():
        for specific_attribute in specific_attributes:
            if (general_attribute != specific_attribute) and (specific_attribute in attribute_groups):
                del new_attribute_groups[specific_attribute]

    return new_attribute_groups, subjects


def generate_rules(
    attribute_groups: Dict[str, List[str]],
    subject: str,
    subject_category: str,
    max_rules: int = 5,
    specific_category: bool = False,
) -> List[MELTLanguageRule]:
    """Generates a random set of rules about a subject as dictionaries,
    given a list of potential attributes and the category (e.g. người) of the subject (e.g. An)

    These rules are guaranteed to not contradict one another, and attributes implied by a single rule will
    not imply any attributes in any other rules (i.e. there is only a single step of reasoning).
    """
    attributes_shuffled = list(attribute_groups.keys()).copy()
    random.shuffle(attributes_shuffled)
    rules: List[MELTLanguageRule] = []

    while len(attributes_shuffled) > 2 and len(rules) < max_rules:
        rule_subject = subject if specific_category else random.choice([subject_category, subject])
        n_rule_attributes = random.randint(2, 3)
        rule_attributes, attributes_shuffled = (
            attributes_shuffled[:n_rule_attributes],
            attributes_shuffled[n_rule_attributes:],
        )
        rules.append(
            MELTLanguageRule(
                subject=rule_subject,
                subject_category=subject_category,
                specifier_type="một",
                condition=rule_attributes[:-1],
                condition_conjunction=random.choice(["và", "hoặc"]),
                consequent=rule_attributes[-1],
            )
        )
    return rules


def generate_test(
    attribute_groups: Dict[str, List[str]],
    subject: str,
    subject_category: str,
    rules: List[MELTLanguageRule],
    use_specific_attributes: bool,
    p_consequenceless=0.1,
) -> Tuple[MELTLanguageFact, List[MELTLanguageRule], MELTLanguageFact]:
    """Generates a test case given a set of rules, i.e. a statement about the subject from which something
    can be potentially deduced given the rules. We include an argument, p_consequenceless, to re-roll with
    some probability if the generated fact does not allow anything to be determined.
    """

    # The generic attributes which the test fact will assign to the subject
    test_attributes: List[str] = random.sample(list(attribute_groups.keys()), 2)
    # The specific versions of the test attributes
    test_attributes_specific: List[str] = [
        random.choice(attribute_groups[subcondition]) for subcondition in test_attributes
    ]
    test_consequents: List[str] = []  # The attributes implied by the test attributes and rules
    test_rules_used: List[MELTLanguageRule] = []
    for rule in rules:
        if rule.consequent in test_attributes:
            continue
        if rule.condition_conjunction == "và":
            if set(rule.condition).issubset(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule.consequent)
        elif rule.condition_conjunction == "hoặc":
            if not set(rule.condition).isdisjoint(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule.consequent)
    if len(test_consequents) == 0 and random.random() > p_consequenceless:
        return generate_test(
            attribute_groups, subject, subject_category, rules, use_specific_attributes, p_consequenceless
        )

    test_fact: MELTLanguageFact = MELTLanguageFact(
        subject,
        subject_category,
        specifier_type="cái",
        specific_attributes=test_attributes_specific,
        generic_attributes=test_attributes,
        use_specific_attributes=use_specific_attributes,
    )

    target_fact: MELTLanguageFact = dataclasses.replace(
        test_fact,
        specific_attributes=test_consequents,
        generic_attributes=test_consequents,
    )

    return test_fact, test_rules_used, target_fact


class MELTSRNScenario(Scenario):
    """
    Synthetic Reasoning Natural Language benchmark inspired by "Transformers as Soft Reasoners over Language"
        https://arxiv.org/abs/2002.05867
    """

    name = "sythetic_reasoning_natural"
    description = "Language Pattern Matching"
    tags = ["reasoning", "language", "pattern_matching"]

    def __init__(self, difficulty: str, random_seed=42):
        super().__init__()
        self.attribute_groups, self.subjects = get_vocab()

        # specific_category specifies that the specific category should always be used
        # e.g. "dog" instead of "an động vật"
        self.specific_category: bool = difficulty == "easy"
        # use_specific_attributes specifies that the synonymous attributes can be used
        # e.g. "chill" instead of "cold"
        self.use_specific_attributes: bool = difficulty == "hard"
        self.include_intermediates: bool = False
        self.num_train_instances: int = 1000
        self.num_val_instances: int = 5000
        self.num_test_instances: int = 5000
        self.random_seed = random_seed

    def generate_problem(
        self,
    ) -> Tuple[List[MELTLanguageRule], MELTLanguageFact, List[MELTLanguageRule], MELTLanguageFact]:
        subject_category = random.choice(list(self.subjects.keys()))
        subject = random.choice(self.subjects[subject_category])
        rules = generate_rules(
            self.attribute_groups, subject, subject_category, specific_category=self.specific_category
        )
        test_fact, test_rules_used, target_fact = generate_test(
            self.attribute_groups, subject, subject_category, rules, self.use_specific_attributes
        )
        return rules, test_fact, test_rules_used, target_fact

    def get_instances(self, output_path: str) -> List[Instance]:
        # Read all the instances
        instances: List[Instance] = []
        random.seed(self.random_seed)

        for sample_idx in range(self.num_train_instances + self.num_val_instances + self.num_test_instances):
            rules, test_fact, test_rules_used, target_fact = self.generate_problem()

            question = "\n".join(str(rule) for rule in rules) + "\n"
            test_specified_subject = test_fact.generate_specified_subject(upper=False)
            question += f"Sự thật:\n{test_fact}\n"
            if self.include_intermediates:
                question += "Luật đã dùng:\n" + "\n".join(str(test_rule) for test_rule in test_rules_used) + "\n"
            question += f"Những điều sau đây có thể được xác định về {test_specified_subject}:"

            if sample_idx < self.num_train_instances:
                split = TRAIN_SPLIT
            elif sample_idx < self.num_train_instances + self.num_val_instances:
                split = VALID_SPLIT
            else:
                split = TEST_SPLIT

            instance = Instance(
                input=Input(text=question),
                references=[Reference(Output(text=str(target_fact)), tags=[CORRECT_TAG])],
                split=split,
            )
            instances.append(instance)

        return instances


ANIMALS = [
    "con ngựa vằn",
    "con rắn hổ mang",
    "con cò",
    "con chim cánh cụt",
    "con cá mập",
    "con sư tử",
    "con trâu",
    "con cá voi",
    "con hải cẩu",
    "con đại bàng",
    "con ngựa",
    "con chuột",
]
FRUITS = [
    "quả táo",
    "quả đào",
    "quả dưa hấu",
    "quả chuối",
    "quả nho",
    "quả kiwi",
    "quả lê",
    "quả dâu tây",
    "quả việt quất",
    "quả mâm xôi",
]
RULE_SYMBOLS = ["X", "Y", "Z"]
MATH_SYMBOLS = ["+", "-", "*", "="]


def subst(pattern: List[str], rule_symbol: str, substitute_str: str) -> List[str]:
    """
    We substitute one rule symbols in a pattern according by a substitution str.

    example:
    pattern = "A+B=B+A"
    rule_symbol = "A"
    substitute_str = "quả táo"
    return: "quả táo+B=B+quả táo"

    :param pattern: A Pattern representing the rule.
    :param rule_symbol: One rule symbol.
    :param substitute_str: The substitution string.
    :return: The result of substitution.
    """
    assert rule_symbol in pattern
    # check which index is the same as rule_symbol
    indices = [i for i, x in enumerate(pattern) if x == rule_symbol]

    # form a new string with the symbol replaced
    new_string = pattern[: indices[0]] + [substitute_str]
    for i, j in zip(indices[:-1], indices[1:]):
        new_string += pattern[i + 1 : j]
        new_string += [substitute_str]
    new_string += pattern[indices[-1] + 1 :]

    return new_string


def pattern_subst(pattern: List[str], rule_symbols: List[str], substitute_dict: Dict[str, str]) -> List[str]:
    """
    We substitute the rule symbols in a pattern according to a substitution dictionary.

    example:
    pattern = "A+B=B+A"
    rule_symbols = ["A", "B"]
    substitute_dict = {"A":"quả táo", "B":"quả đào"}
    return: "quả táo+quả đào=quả đào+quả táo"

    :param pattern: A Pattern representing the rule.
    :param rule_symbols: The set of rule symbols.
    :param substitute_dict: The substitution dictionary.
    :return: The result of substitution.
    """

    out = pattern
    # we iteratively replace each rule symbol with its subsitution string
    for symbol in rule_symbols:
        out = subst(out, symbol, substitute_dict[symbol])
    return out


class MELTSyntheticReasoningScenario(Scenario):
    """
    Synthetic Reasoning benchmark inspired by
    "LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning"
        https://arxiv.org/abs/2101.06223
    """

    name = "synthetic_reasoning"
    description = "Synthetic reasoning benchmark"
    tags = ["reasoning", "language", "pattern_matching"]

    def __init__(self, mode: str, random_seed=42):
        super().__init__()
        self.num_train_instances: int = 1000
        self.num_val_instances: int = 5000
        self.num_test_instances: int = 5000
        self.rng = np.random.RandomState(random_seed)
        self.mode = mode
        assert self.mode in ["variable_substitution", "pattern_match", "induction"], f"Unsupported mode: {self.mode}"

    def gen_subst(self, rule_symbols: List[str], tokens: List[str]) -> Tuple[Dict[str, str], str]:
        """
        For each of the rule symbol, we sample a random substitution string composed of randomly sampled tokens.

        :param rule_symbols: A list of rule symbols.
        :param tokens: Tokens used to construct the substitution.
        :return: We return a substitution dictionary, and its string representation.
        """
        substitute_dict = {}
        substitute_str = []
        for char in rule_symbols:
            subst_len = self.rng.randint(1, 3)
            subst = " ".join(self.rng.choice(tokens, size=subst_len))
            substitute_dict.update({char: subst})
            substitute_str.append(char)
            substitute_str.append("bởi")
            substitute_str.append('"')
            substitute_str.append(subst)
            substitute_str.append('"')
            substitute_str.append(",")
        substitute_dict_str = " ".join(substitute_str[:-1])
        return substitute_dict, substitute_dict_str

    def gen_pattern(self, math_symbols: List[str], rule_symbols: List[str]) -> List[str]:
        """
        Generate a pattern string.

        Example Input: math_symbols: ["+", "-", "*"], rule_symbols: ["Y", "Y", "Z"]
        Example Output: ["Y", "Y", "+", "Z", "="]
        """
        pattern = rule_symbols + math_symbols
        self.rng.shuffle(pattern)
        return pattern

    def get_instances(self, output_path: str) -> List[Instance]:
        # We fix the seed for data generation to ensure reproducibility.
        # Read all the instances
        instances: List[Instance] = []

        rule_symbols = RULE_SYMBOLS
        tokens = ANIMALS + FRUITS
        math_symbols = MATH_SYMBOLS

        for sample_idx in range(self.num_train_instances + self.num_val_instances + self.num_test_instances):
            # Sample rule symbols
            sampled_rule_symbols = list(self.rng.choice(rule_symbols, size=self.rng.randint(2, 4)))
            sampled_rule_symbols_set = sorted(list(set(sampled_rule_symbols)))  # sorted to make it deterministic

            # Sample math symbols
            sampled_math_symbols = list(self.rng.choice(math_symbols, size=self.rng.randint(2, 4)))

            # generate the pattern
            pattern = self.gen_pattern(sampled_math_symbols, sampled_rule_symbols)

            # generate one substitution
            substitute_dict, substitute_dict_str = self.gen_subst(sampled_rule_symbols_set, tokens)
            result = pattern_subst(pattern, sampled_rule_symbols_set, substitute_dict)

            # generate another substitution
            substitute_dict_2, _ = self.gen_subst(sampled_rule_symbols_set, tokens)
            result_2 = pattern_subst(pattern, sampled_rule_symbols_set, substitute_dict_2)

            result_string = " ".join(result)
            pattern_string = " ".join(pattern)

            src: str
            tgt: str
            if self.mode == "induction":
                result_string_2 = " ".join(result_2)
                src = f"Hai kết quả: {result_string} | {result_string_2}"
                tgt = f"Quy luật: {pattern_string}"
            elif self.mode == "variable_substitution":
                src = f"Các quy luật: {pattern_string} | Thay thế: {substitute_dict_str}"
                tgt = " ".join(result)
            elif self.mode == "pattern_match":
                # we sample 3 other pattern strings as negatives for patterns matching.
                other_patterns = [
                    " ".join(self.gen_pattern(sampled_math_symbols, sampled_rule_symbols_set)) for _ in range(3)
                ]
                all_patterns = other_patterns + [pattern_string]
                self.rng.shuffle(all_patterns)
                all_pattern_string = " | ".join(all_patterns)
                src = f"Các quy luật: {all_pattern_string} | Kết quả: {result_string}"
                tgt = pattern_string
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            split: str
            if sample_idx < self.num_train_instances:
                split = TRAIN_SPLIT
            elif sample_idx < self.num_train_instances + self.num_val_instances:
                split = VALID_SPLIT
            else:
                split = TEST_SPLIT

            instance = Instance(
                input=Input(text=src),
                references=[Reference(Output(text=tgt), tags=[CORRECT_TAG])],
                split=split,
            )
            instances.append(instance)

        return instances


def remove_boxed(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _fix_fracs(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions.

    Examples:
    >>> _fix_fracs("\\frac1b")
    \frac{1}{b}
    >>> _fix_fracs("\\frac12")
    \frac{1}{2}
    >>> _fix_fracs("\\frac1{72}")
    \frac{1}{72}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions formatted as a/b to \\frac{a}{b}.

    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Remove units (on the right).
    "\\text{ " only ever occurs (at least in the val set) when describing units.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat square roots.

    Example:
    >>> _fix_sqrt("\\sqrt3")
    \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Apply the reformatting helper functions above.
    """
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def get_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer


def is_equiv(str1: Optional[str], str2: Optional[str]) -> float:
    """Returns (as a float) whether two strings containing math are equivalent up to differences of formatting in
    - units
    - fractions
    - square roots
    - superfluous LaTeX.

    Source: https://github.com/hendrycks/math
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return float(ss1 == ss2)
    except Exception:
        return float(str1 == str2)


def is_equiv_chain_of_thought(str1: str, str2: str) -> float:
    """Strips the solution first before calling `is_equiv`."""
    ans1 = get_answer(str1)
    ans2 = get_answer(str2)

    return is_equiv(ans1, ans2)


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
