"""Synthetic Reasoning Natural Language Scenario.

We define a set of reasoning tasks related to pattern matching in natural language. In essence, each problem is composed
of some combination of

- Rules, a list of conditional statements such as "If a person is red and kind, then the person is cold."
- Fact, a single case from which something may or may not be deduced given the rules.
    For example, "The dog is big and red."
- Consequents, the set of all things implied by the combination of the fact and rules.
    For example, given a problem such as

        Rules:
        If a cow is weak, then the cow is small.
        If a cow is hot, then the cow is purple.
        If a cow is beautiful and slow, then the cow is bad.
        If a cow is old, then the cow is cold.
        If a cow is green and red, then the cow is strong.
        Fact:
        A cow is smart and hot.
        The following can be determined about the cow:

    The consequent would be "The cow is purple."
- Intermediates used, the set of rules which are actually used to go from the rules and fact to the consequent.
    In the previous example, this would be "If a cow is hot, then the cow is purple"

We can support a variety of tasks from this framework.

- Rules + Fact -> Consequents (highlights deduction)
- Intermediates + Consequents -> Fact (abduction)
- Facts + Consequents -> Intermediates (induction)
- Rules + Fact -> Intermediates + Consequents (a variation on the first example with intermediate steps)
- Rules + Fact -> Intermediates (a pure pattern matching test, without substitution)

We also support multiple levels of difficulty.

- At the easy level, we assume that the subject and any attributes match exactly in any rules and facts
- At the medium level, we add the need to understand that the subject of rules may be a broader class
    For example, instead of

        "If Carol is happy, then Carol is green."

    We may have

        "If a person is happy, then the person is green."

    And the model would need to still apply this rule to Carol.
- At the hard level, we add the need to understand that the attributes of rules may be a broader class
    (In addition to the subject abstraction from the medium level.)
    For example, consider the rule:

        "If an animal is cold or old, then the animal is good."

    Instead of

        "The dog is old and big."

    We may have

        "The dog is ancient and huge."

    And the model would need to still apply this rule to Carol.
"""

import random
import dataclasses
from copy import copy
from typing import List, Dict, Literal, Tuple
from dataclasses import dataclass

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


@dataclass(frozen=True)
class LanguageLogicalStatement:
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
class LanguageRule(LanguageLogicalStatement):
    """Class describing how a set of attributes about an individual/group imply another attribute."""

    condition: List[str]  # a list of attributes which must apply for the rule to apply
    condition_conjunction: Literal["and", "or"]  # "and" or "or", corresponding to
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
        return f"If {specified_subject} is {condition}, then {specified_particular_subject} is {self.consequent}."


@dataclass(frozen=True)
class LanguageFact(LanguageLogicalStatement):
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
            return "Nothing."
        target_attributes = self.specific_attributes if self.use_specific_attributes else self.generic_attributes
        specified_subject = self.generate_specified_subject(upper=self.upper)
        return f"{specified_subject} is {' and '.join(target_attributes)}."


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
        "person": ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank"],
        "animal": [
            "dog",
            "cat",
            "rabbit",
            "mouse",
            "tiger",
            "lion",
            "bear",
            "squirrel",
            "cow",
            "panda",
            "hedgehog",
            "elephant",
            "giraffe",
            "hippo",
        ],
        "plant": ["poppy", "dandelion", "tree", "rose", "sunflower"],
    }

    # Convert list of attributes into dictionary
    # A list of attributes and their overarching meaning (used in hard difficulty)
    attribute_groups = {
        "young": ["young"],
        "soft": ["soft"],
        "sad": ["sad"],
        "scary": ["scary"],
        "cold": ["cold", "chill", "cool"],
        "hot": ["hot", "warm"],
        "smart": ["smart", "clever", "wise", "intelligent"],
        "clean": ["clean", "tidy"],
        "small": ["small", "little", "tiny"],
        "big": ["big", "enormous", "giant", "huge"],
        "good": ["good", "kind", "nice"],
        "beautiful": ["beautiful", "pretty"],
        "red": ["red", "crimson"],
        "blue": ["blue", "cobalt"],
        "green": ["green", "viridian"],
        "purple": ["purple", "violet"],
        "boring": ["boring", "dull"],
        "old": ["old", "ancient", "antique"],
        "strong": ["strong", "powerful", "muscular"],
        "weak": ["weak", "frail", "fragile"],
        "fast": ["fast", "quick"],
        "slow": ["slow", "sluggish"],
        "bad": ["bad", "evil", "wicked", "mean"],
        "happy": ["happy", "elated", "glad", "cheerful"],
        "round": ["round", "circular", "spherical"],
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
) -> List[LanguageRule]:
    """Generates a random set of rules about a subject as dictionaries,
    given a list of potential attributes and the category (e.g. person) of the subject (e.g. Alice)

    These rules are guaranteed to not contradict one another, and attributes implied by a single rule will
    not imply any attributes in any other rules (i.e. there is only a single step of reasoning).
    """
    attributes_shuffled = list(attribute_groups.keys()).copy()
    random.shuffle(attributes_shuffled)
    rules: List[LanguageRule] = []

    while len(attributes_shuffled) > 2 and len(rules) < max_rules:
        rule_subject = subject if specific_category else random.choice([subject_category, subject])
        n_rule_attributes = random.randint(2, 3)
        rule_attributes, attributes_shuffled = (
            attributes_shuffled[:n_rule_attributes],
            attributes_shuffled[n_rule_attributes:],
        )
        rules.append(
            LanguageRule(
                subject=rule_subject,
                subject_category=subject_category,
                specifier_type="a",
                condition=rule_attributes[:-1],
                condition_conjunction=random.choice(["and", "or"]),
                consequent=rule_attributes[-1],
            )
        )
    return rules


def generate_test(
    attribute_groups: Dict[str, List[str]],
    subject: str,
    subject_category: str,
    rules: List[LanguageRule],
    use_specific_attributes: bool,
    p_consequenceless=0.1,
) -> Tuple[LanguageFact, List[LanguageRule], LanguageFact]:
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
    test_rules_used: List[LanguageRule] = []
    for rule in rules:
        if rule.consequent in test_attributes:
            continue
        if rule.condition_conjunction == "and":
            if set(rule.condition).issubset(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule.consequent)
        elif rule.condition_conjunction == "or":
            if not set(rule.condition).isdisjoint(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule.consequent)
    if len(test_consequents) == 0 and random.random() > p_consequenceless:
        return generate_test(
            attribute_groups, subject, subject_category, rules, use_specific_attributes, p_consequenceless
        )

    test_fact: LanguageFact = LanguageFact(
        subject,
        subject_category,
        specifier_type="the",
        specific_attributes=test_attributes_specific,
        generic_attributes=test_attributes,
        use_specific_attributes=use_specific_attributes,
    )

    target_fact: LanguageFact = dataclasses.replace(
        test_fact,
        specific_attributes=test_consequents,
        generic_attributes=test_consequents,
    )

    return test_fact, test_rules_used, target_fact


class SRNScenario(Scenario):
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

    def generate_problem(self) -> Tuple[List[LanguageRule], LanguageFact, List[LanguageRule], LanguageFact]:
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
            question += f"Fact:\n{test_fact}\n"
            if self.include_intermediates:
                question += "Rule(s) used:\n" + "\n".join(str(test_rule) for test_rule in test_rules_used) + "\n"
            question += f"The following can be determined about {test_specified_subject}:"

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
