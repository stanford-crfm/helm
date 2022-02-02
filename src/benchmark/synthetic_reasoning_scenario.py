import random
from copy import copy
from typing import List
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


ANIMALS = ["zebra", "cobra", "stork", "penguin", "shark", "lion", "buffalo", "whale", "seal", "eagle", "horse", "rat"]
FRUITS = ["apple", "peach", "watermelon", "banana", "grape", "kiwi", "pear", "strawberry", "blueberry", "blackberry"]
RULE_SYMBOLS = ["X", "Y", "Z"]
MATH_SYMBOLS = ["+", "-", "*", "="]


def sample_substring(string, min, max):
    substring_len = random.randint(min, max)
    init = random.randint(0, len(string) - substring_len)
    return string[init:init + substring_len], init, substring_len


class Pattern:
    def __init__(self, pattern_list):
        self.pattern_list = pattern_list

    def replace(self, to_replace, replace):
        indices = []
        for i, char in enumerate(self.pattern_list):
            if char == to_replace:
                indices.append(i)

        new_pattern = []
        new_pattern.extend(self.pattern_list[:indices[0]])
        if not isinstance(replace, list):
            replace = [replace]
        new_pattern.extend(replace)
        for i, j in zip(indices[:-1], indices[1:]):
            new_pattern.extend(self.pattern_list[i+1:j])
            new_pattern.extend(replace)
        new_pattern.extend(self.pattern_list[indices[-1]+1:])
        return new_pattern


class SubstRule:
    '''
    example:
    addcomm = SubstRule(["A", "B"], "A+B=B+A")
    out = addcomm.rewrite({"A":"xwd239+c", "B":"23oijs_c"})
    xwd239+c+23oijs_c=23oijs_c+xwd239+c
    '''
    def __init__(self, upper_case_letters, pattern):
        self.upper_case_letters = upper_case_letters
        self.pattern = Pattern(pattern)
    def subst(self, substitute):
        out = self.pattern
        for char in self.upper_case_letters:
            out = Pattern(out.replace(char, substitute[char]))
        return out.pattern_list


def gen_subst(upper_case_letters, lower_case_letters):
    substitute = {}
    substitute_str = []
    for char in upper_case_letters:
        subst_len = random.randint(1, 2)
        subst = random.choices(lower_case_letters,
                               k=subst_len)
        substitute.update({char: subst})
        substitute_str.append(char)
        substitute_str.append("by")
        substitute_str.append('"')
        substitute_str.extend(subst)
        substitute_str.append('"')
        substitute_str.append(",")
    return substitute, substitute_str[:-1]


class SyntheticReasoningScenario(Scenario):
    """
    Language Pattern Matching benchmark based on "Transformers as Soft Reasoners over Language"
        https://arxiv.org/abs/2002.05867
    """

    name = "lpm"
    description = "Language Pattern Matching"
    tags = ["reasoning", "language", "pattern_matching"]

    def __init__(self, mode: str, random_seed=42):
        self.n_train_samples = 5
        self.n_valid_samples = 50
        self.n_test_samples = 50
        self.mode = mode
        assert self.mode in ["variable_substitution", "pattern_match", "induction"]
        random.seed(random_seed)

    def get_instances(self) -> List[Instance]:
        # Read all the instances
        instances = []

        for sample_idx in range(self.n_train_samples + self.n_valid_samples + self.n_test_samples):
            upper_case_letters = RULE_SYMBOLS
            lower_case_letters = ANIMALS + FRUITS
            math_symbols = MATH_SYMBOLS

            num_chars = random.randint(1, 3)
            pattern_length = random.choice(range(2, 4))
            upper_case_letters = upper_case_letters[:num_chars]
            pattern_chars = random.choices(upper_case_letters, k=random.choice(range(1, 4)))
            upper_case_letters = list(set(pattern_chars))
            pattern = pattern_chars + random.choices(math_symbols, k=pattern_length)
            random.shuffle(pattern)
            rule = SubstRule(upper_case_letters, pattern)
            substitute, substitute_str = gen_subst(upper_case_letters, lower_case_letters)
            result = rule.subst(substitute)
            substitute_2, substitute_str_2 = gen_subst(upper_case_letters, lower_case_letters)
            result_2 = rule.subst(substitute_2)

            if self.mode == "induction":
                src = "Two results:" + " ".join([str(c) for c in result + ["|"] + result_2])
                tgt = "Rule:" + " ".join([str(c) for c in pattern])
            elif self.mode == "variable_substitution":
                src = " ".join([str(c) for c in ["Rule:"] + pattern + ["|"] + ["Substitution:"] + substitute_str])
                tgt = " ".join([str(c) for c in result])
            elif self.mode == "pattern_match":
                src = " ".join([str(c) for c in ["Rule:"] + pattern + ["|"] + ["Result:"] + result])
                tgt = " ".join([str(c) for c in ["Substitution:"] + substitute_str])
            else:
                raise ValueError("Mode {} not found".format(self.mode))

            if sample_idx < self.n_train_samples:
                cur_tag = TRAIN_TAG
            elif sample_idx < self.n_train_samples + self.n_valid_samples:
                cur_tag = VALID_TAG
            else:
                cur_tag = TEST_TAG
            instance = Instance(
                input=src, references=[Reference(output=tgt, tags=[CORRECT_TAG])], tags=[cur_tag],
            )
            instances.append(instance)

        return instances
