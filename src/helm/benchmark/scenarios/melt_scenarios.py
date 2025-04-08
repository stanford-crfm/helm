import os
import json
import pickle
import dataclasses
import random
import numpy as np

from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

from datasets import load_dataset
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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
        passage_prefix: Optional[str] = "Passage: ",
        question_prefix: Optional[str] = "Question: ",
        splits: Dict[str, str] = None,
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
        article_key: Optional[str] = "source",
        summary_key: Optional[str] = "target",
        splits: Dict[str, str] = None,
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
    """

    name = "melt_summarization_vietnews"
    description = "Vietnews is a Vietnamese summarization dataset."
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
    specifier_type: Literal["a", "the"]  # the specifier used for the subject

    def generate_specified_subject(self, upper=False, specifier_type=None) -> str:
        """Figure out whether to use "A" or "An" (heuristic), handles "the" and "The" too
        specifier_type can be used to optionally override the class property specifier_type

        Example:
            if (subject="cat", subject_category="animal", specifier_type="the", upper=False) -> "the cat"
            if (subject="apple", subject_category="plant", specifier_type="a", upper=True) -> "An apple"
        """

        specifier_type = self.specifier_type if specifier_type is None else specifier_type
        if not (self.subject_category != "person") or (self.subject == "person"):
            return self.subject
        base_char = specifier_type[0].upper() if upper else specifier_type[0].lower()
        if specifier_type == "a":
            if self.subject[0].lower() in ["a", "e", "i", "o", "u"]:
                return f"{base_char}n {self.subject}"
        else:
            return f"{base_char}{specifier_type[1:]} {self.subject}"
        return f"{base_char} {self.subject}"


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
            'subject': 'Alice',
            'subject_category': 'person',
            'specifier_type': 'the' or 'a'
            'condition': ['red', 'kind'],
            'condition_conjunction': 'and',
            'consequent': 'cold'
        }

        and this example will output a string: "If Alice is red and kind, then Alice is cold."
        """

        condition = f" {self.condition_conjunction} ".join(self.condition)
        specified_subject = self.generate_specified_subject()
        specified_particular_subject = self.generate_specified_subject(specifier_type="the")
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

        e.g. if (subject="dog", attributes=["big", "red"], specifier="The") ->
        "The dog is big and red."
        """

        if len(self.generic_attributes) == 0:
            return "Không có gì."
        target_attributes = self.specific_attributes if self.use_specific_attributes else self.generic_attributes
        specified_subject = self.generate_specified_subject(upper=self.upper)
        return f"{specified_subject} là {' và '.join(target_attributes)}."


def get_vocab() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """All potential subjects for the facts and rules for sythetic_reasoning_natural as well as their categories.
    Subjects is a dictionary of subject categories like "person" and "animal" which correspond to
    a list of potential subjects.

    Attributes corresponds to an initial list of attributes which are only synonymous with themselves.
    Intially, we default to not including these attributes, but we leave this feature in for convenience.

    Attribute groups are a more general version of attributes, where a single attribute corresponds to a class of
    attributes e.g. if we know something is chilly, we know that it is cold (but not assuming the reverse).
    """

    # A list of subjects and their categories
    subjects: Dict[str, List[str]] = {
        "person": ["An", "Bình", "Cường", "Duy", "Đạt", "Phương"],
        "animal": [
            "chó",
            "mèo",
            "thỏ",
            "chuột",
            "hổ",
            "sư tử",
            "gấu",
            "sóc",
            "bò",
            "gấu trúc",
            "nhím",
            "voi",
            "hươu cao cổ",
            "hà mã",
        ],
        "plant": ["anh túc", "bồ công anh", "cây", "hoa hồng", "hoa hướng dương"],
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
        "thông minh": ["thông minh", "giỏi", "không", "sáng trí"],
        "sạch": ["sạch", "ngăn nắp"],
        "nhỏ": ["nhỏ", "nhỏ xíu", "tí nị"],
        "to": ["to", "khổng lồ", "to lớn", "lớn"],
        "tốt": ["tốt", "tử tế", "tốt bụng"],
        "đẹp": ["đẹp", "xinh"],
        "đỏ": ["đỏ", "đỏ thẫm"],
        "xanh dưogn": ["xanh dương", "xanh lam"],
        "xanh lục": ["xanh lục", "xanh lá cây"],
        "tím": ["tím", "tím than"],
        "chán": ["chán", "đần"],
        "cũ": ["cũ", "xưa", "cổ"],
        "mạnh": ["mạnh", "mạnh mẽ", "cơ bắp"],
        "yếu": ["yếu", "yếu đuối", "mỏng manh"],
        "nhanh": ["nhanh", "mau"],
        "chậm": ["chậm", "chậm chạp"],
        "dở": ["dở", "tệ", "ác", "đê tiện"],
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
    given a list of potential attributes and the category (e.g. person) of the subject (e.g. Alice)

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
                specifier_type="a",
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
        specifier_type="the",
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
        # e.g. "dog" instead of "an animal"
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
    "ngựa vằn",
    "rắn hổ mang",
    "cò",
    "chim cánh cụt",
    "cá mập",
    "sư tử",
    "trâu",
    "cá voi",
    "hải cẩu",
    "đại bàng",
    "ngựa",
    "chuột",
]
FRUITS = ["táo", "đào", "dưa hấu", "chuối", "nho", "kiwi", "lê", "dâu tây", "việt quất", "mâm xôi"]
RULE_SYMBOLS = ["X", "Y", "Z"]
MATH_SYMBOLS = ["+", "-", "*", "="]


def subst(pattern: List[str], rule_symbol: str, substitute_str: str) -> List[str]:
    """
    We substitute one rule symbols in a pattern according by a substitution str.

    example:
    pattern = "A+B=B+A"
    rule_symbol = "A"
    substitute_str = "apple"
    return: "apple+B=B+apple"

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
    substitute_dict = {"A":"apple", "B":"peach"}
    return: "apple+peach=peach+apple"

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
                tgt = f"Luật: {pattern_string}"
            elif self.mode == "variable_substitution":
                src = f"Các luật: {pattern_string} | Substitutions: {substitute_dict_str}"
                tgt = " ".join(result)
            elif self.mode == "pattern_match":
                # we sample 3 other pattern strings as negatives for patterns matching.
                other_patterns = [
                    " ".join(self.gen_pattern(sampled_math_symbols, sampled_rule_symbols_set)) for _ in range(3)
                ]
                all_patterns = other_patterns + [pattern_string]
                self.rng.shuffle(all_patterns)
                all_pattern_string = " | ".join(all_patterns)
                src = f"Các luật: {all_pattern_string} | Kết quả: {result_string}"
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
