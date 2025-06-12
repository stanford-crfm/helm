from typing import Dict, List, Tuple, Literal

import random
import dataclasses

from copy import copy
from dataclasses import dataclass
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


@dataclass(frozen=True)
class MELTLanguageLogicalStatement:
    """This class describes a logical statement in Vietnamese language, inspired by HELM
    implementation of "Transformers as Soft Reasoners over Language" paper.
    https://arxiv.org/abs/2002.05867
    """

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
    """Class describing how a set of attributes about an individual/group imply another attribute.
    This class is inspired by HELM
    implementation of "Transformers as Soft Reasoners over Language" paper.
    https://arxiv.org/abs/2002.05867
    """

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
    """Class describing a statement that a subject has some attributes.
    This class is inspired by HELM
    implementation of "Transformers as Soft Reasoners over Language" paper.
    https://arxiv.org/abs/2002.05867
    """

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
