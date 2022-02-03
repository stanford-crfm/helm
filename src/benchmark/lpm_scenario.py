import random
from copy import copy
from typing import List
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


def get_vocab():
    """All potential subjects for the facts and rules for LPM as well as their categories. 
    Subjects is a dictionary of subject categories like "person" and "animal" which correspond to
    a list of potential subjects. 
    
    Attributes corresponds to an initial list of attributes which are only synonymous with themselves.
    Intially, we default to not including these attributes, but we leave this feature in for convenience.
    
    Attribute groups are a more general version of attributes, where a single attribute corresponds to a class of 
    attributes e.g. if we know something is chilly, we know that it is cold (but not assuming the reverse.
    """

    # A list of subjects and their categories
    subjects = {
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

    # A list of attributes for the facts and rules which only correspond to themselves
    attributes = [
        # 'red', 'blue', 'green', 'young', 'rough',
        # 'fuzzy', 'sad', 'tasty', 'soft', 'fine', 'funny', 'rich',
        # 'new', 'sick', 'poor', 'boring', 'powerful',
        # 'friendly', 'orange', 'wide', 'lost', 'yellow',
        # 'white', 'grey', 'black', 'pink', 'brown', 'cheerful',
        # 'striped', 'wild', 'shiny',
        # 'fluffy', 'gray', 'curly', 'busy', 'plain', 'flat',  'sleek',
        # 'scary', 'harmless', 'graceful'
        # # 'heavy', 'tall', 'short', 'light', 'long', 'fat', 'gentle', 'elegant', 'sparkly', 'great',
    ]

    # Convert list of attributes into dictionary
    attribute_groups = {attribute: [attribute] for attribute in attributes}
    # A list of attributes and their overarching meaning (used in hard difficulty)
    attribute_groups.update(
        {
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
            "happy": ["happy", "elated", "glad"],
            "round": ["round", "circular", "spherical"],
        }
    )
    # Remove any keys which duplicate subitems
    new_attribute_groups = copy(attribute_groups)
    for general_attribute, specific_attributes in attribute_groups.items():
        for specific_attribute in specific_attributes:
            if (general_attribute != specific_attribute) and (specific_attribute in attribute_groups):
                del new_attribute_groups[specific_attribute]
    attribute_groups = new_attribute_groups

    return attribute_groups, subjects


def generate_specifier(subject, upper=False):
    """Figure out whether to use "A" or "An" (heuristic)

    This determines whether to use A or An in a given sentence for a subject. 

    Example:
        generate_specifier("cat", upper=False) -> "a"
        generate_specifier("apple", upper=True) -> "An"
    """

    base_char = "A" if upper else "a"
    if subject[0].lower() in ["a", "e", "i", "o", "u"]:
        return base_char + "n"
    return base_char


def parse_rule(rule):
    """Takes in a rule as a dictionary, i.e. corresponding to "if x (and/or y) then z" 
    and returns a string corresponding to the rule

    Rules should have the following format:
    {
        'subject': 'person',
        'condition': ['red', 'kind'],
        'condition_conjunction': 'and',
        'consequent': 'cold'
        'specified': True
    }
    
    and this example will output a string: "If a person is red and kind, then the person is cold."
    """

    condition = f" {rule['condition_conjunction']} ".join(rule["condition"])
    specified = rule["specified"]
    if specified:
        specifier = generate_specifier(rule["subject"])
        return f"If {specifier} {rule['subject']} is {condition}, then the {rule['subject']} is {rule['consequent']}."
    return f"If {rule['subject']} is {condition}, then {rule['subject']} is {rule['consequent']}."


def parse_fact(subject, attributes, prefix="", specifier=""):
    """Maps from a set of attributes about a subject to a string

    e.g. parse_fact(subject="dog", attributes=["big", "red"], specifier="The") ->
    "The dog is big and red."
    """

    if len(attributes) == 0:
        return "Nothing."
    return f"{prefix}{specifier}{subject} is {' and '.join(attributes)}."


def generate_rules(attribute_groups, subject_category, subject, max_rules=5, specific_category=False):
    """ Generates a random set of rules about a subject as dictionaries, 
    given a list of potential attributes and the category (e.g. person) of the subject (e.g. Alice)
    
    These rules are guaranteed to not contradict one another, and attributes implied by a single rule will 
    not imply any attributes in any other rules (i.e. there is only a single step of reasoning).
    """
    attributes_shuffled = list(attribute_groups.keys()).copy()
    random.shuffle(attributes_shuffled)
    rules = []
    hlog("Generating language pattern matching examples")
    while len(attributes_shuffled) > 2 and len(rules) < max_rules:
        rule_subject_category = subject if specific_category else random.choice([subject_category, subject])
        n_rule_attributes = random.randint(2, 3)
        rule_attributes, attributes_shuffled = (
            attributes_shuffled[:n_rule_attributes],
            attributes_shuffled[n_rule_attributes:],
        )
        condition = rule_attributes[:-1]
        condition_conjunction = random.choice(["and", "or"])
        consequent = rule_attributes[-1]
        specied = (subject_category != "person") or (rule_subject_category == "person")
        rules.append(
            {
                "subject": rule_subject_category,
                "condition": condition,
                "condition_conjunction": condition_conjunction,
                "consequent": consequent,
                "specified": specied,
            }
        )
    return rules


def generate_test(attribute_groups, subject, rules, p_consquenceless=0.1):
    """ Generates a test case given a set of rules, i.e. a statement about the subject from which something
    can be potentially deduced given the rules. We include an argument, p_consquenceless, to re-roll with 
    some probability if the generated fact does not allow anything to be determined.
    """
    # test_attributes = random.sample(attributes, 2)
    test_attributes = random.sample(list(attribute_groups.keys()), 2)
    test_attributes_specific = [random.choice(attribute_groups[subcondition]) for subcondition in test_attributes]
    test_consequents = []
    test_rules_used = []
    for rule in rules:
        condition_conjunction = rule["condition_conjunction"]
        rule_condition = rule["condition"]
        rule_consequent = rule["consequent"]
        if rule_consequent in test_attributes:
            continue
        if condition_conjunction == "and":
            if set(rule_condition).issubset(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule_consequent)
        elif condition_conjunction == "or":
            if not set(rule_condition).isdisjoint(test_attributes):
                test_rules_used.append(rule)
                test_consequents.append(rule_consequent)
    if len(test_consequents) == 0 and random.random() > p_consquenceless:
        return generate_test(attribute_groups, subject, rules, p_consquenceless)
    return test_attributes, test_attributes_specific, test_consequents, test_rules_used


class LPMScenario(Scenario):
    """
    Language Pattern Matching benchmark inspired by "Transformers as Soft Reasoners over Language"
        https://arxiv.org/abs/2002.05867
    """

    name = "lpm"
    description = "Language Pattern Matching"
    tags = ["reasoning", "language", "pattern_matching"]

    def __init__(self, difficulty: str, random_seed=42):
        self.attribute_groups, self.subjects = get_vocab()
        # specific_category specifies that the specific category should always be used
        # e.g. "dog" instead of "an animal"
        self.specific_category = difficulty == "easy"
        # generic_attributes specifies that the top level attribute should always be used
        # e.g. "cold" instead of "chill"
        self.generic_attributes = difficulty != "hard"
        self.include_intermediates = False
        self.n_train_samples = 5
        self.n_valid_samples = 50
        self.n_test_samples = 50
        random.seed(random_seed)

    def get_instances(self) -> List[Instance]:
        # Read all the instances
        instances = []

        for sample_idx in range(self.n_train_samples + self.n_valid_samples + self.n_test_samples):
            subject_category = random.choice(list(self.subjects.keys()))
            subject = random.choice(self.subjects[subject_category])
            rules = generate_rules(
                self.attribute_groups, subject_category, subject, specific_category=self.specific_category
            )
            test_attributes, test_attributes_specific, test_consequents, test_rules_used = generate_test(
                self.attribute_groups, subject, rules
            )

            # question = "Rules:\n"
            question = ""
            for rule in rules:
                question += parse_rule(rule) + "\n"

            test_specifier_base = generate_specifier(subject, upper=True)
            test_specifier_first = test_specifier_base + " " if subject_category != "person" else ""
            test_specifier_second = "The " if subject_category != "person" else ""
            test_specifier_third = "the " if subject_category != "person" else ""
            print_test_attributes = test_attributes if self.generic_attributes else test_attributes_specific
            question += "Fact:\n"
            question += parse_fact(subject, print_test_attributes, specifier=test_specifier_first) + "\n"
            if self.include_intermediates:
                question += "Rule(s) used:\n"
                for rule in test_rules_used:
                    question += parse_rule(rule) + "\n"
            question += f"The following can be determined about {test_specifier_third}{subject}:\n"
            correct_answer = parse_fact(subject, test_consequents, specifier=test_specifier_second)

            if sample_idx < self.n_train_samples:
                cur_tag = TRAIN_TAG
            elif sample_idx < self.n_train_samples + self.n_valid_samples:
                cur_tag = VALID_TAG
            else:
                cur_tag = TEST_TAG
            instance = Instance(
                input=question, references=[Reference(output=correct_answer, tags=[CORRECT_TAG])], tags=[cur_tag],
            )
            instances.append(instance)

        return instances
